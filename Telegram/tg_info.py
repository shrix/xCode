import os
import re
import sys
import time
import glob
import asyncio
import subprocess
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument, Dialog
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from telethon.errors import FloodWaitError
import tempfile
from datetime import datetime, timedelta

load_dotenv()

MAX_RETRIES = 5

api_id = int(os.environ["TELEGRAM_API_ID"])
api_hash = os.environ["TELEGRAM_API_HASH"]


# Function to validate content type
def validate_content_type(input_type):
    valid_types = {
        "l": "links",
        "t": "text",
        "i": "images",
        "v": "videos"
    }
    return valid_types.get(input_type.lower())


# Function to extract specified content from message text
async def extract_content(message):
    try:
        content = {
            "links": re.findall(r'(https?://[^\s]+)', message.text) if message.text else [],
            "text": [message.text] if message.text else [],
            "images": [],
            "videos": []
        }

        if message.media:
            if isinstance(message.media, MessageMediaPhoto):
                media_type = "images"
            elif isinstance(message.media, MessageMediaDocument) and message.media.document.mime_type.startswith('video'):
                media_type = "videos"
            else:
                media_type = None

            if media_type:
                try:
                    path = await message.download_media(file=f"data/{media_type}/")
                    content[media_type].append(os.path.abspath(path))
                except Exception as e:
                    print(f"Error downloading media: {e}")

        return content
    except AttributeError as e:
        print(f"Error extracting content: {e}")
        return {}


# Function to fetch and extract content from specified chat
async def fetch_content_from_chat(client):
    try:
        chat_name = input("\nEnter name of Chat/Group/Channel: ").strip()
        content_type = validate_content_type(input("What to extract? ([L]inks / [T]ext / [I]mages / [V]ideos): ").strip().lower())
        if content_type is None:
            print("Invalid content type. Please choose from L, T, I, or V.")
            return
        limit = int(input(f"How many {content_type} to extract? "))

        dialogs = await client.get_dialogs()
        target_chat = next((dialog for dialog in dialogs if dialog.name.lower() == chat_name.lower()), None)

        if not target_chat:
            print(f"'{chat_name}' not found. Please check the name and try again.")
            return

        print(f"\nFetching messages from '{target_chat.name}' ...")

        all_content = []
        message_count = 0
        progress_interval = max(limit // 10, 20)  # Report progress every 10% or 20 messages, whichever is larger

        async for message in client.iter_messages(target_chat, limit=None):
            message_count += 1
            content = await extract_content(message)
            if content.get(content_type):
                all_content.extend(content[content_type])
                print(f"Found {content_type} in message {message.id}")
                if len(all_content) >= limit:
                    break
            if message_count % progress_interval == 0:
                print(f"Checked {message_count} messages ... Found {len(all_content)} {content_type} so far.")

        all_content = all_content[:limit]

        content_folder = {
            'links': 'link',
            'text': 'text',
            'images': 'image',
            'videos': 'video'
        }[content_type]
        output_folder = os.path.join('data', content_folder)
        os.makedirs(output_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chat_name.replace(' ', '_')}_{content_type}_{timestamp}"

        # Save content based on type (text or media)
        if content_type in ['links', 'text']:
            output_file = os.path.join(output_folder, f"{filename}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in all_content:
                    f.write(f"{item}\n")
            print(f"\nSaved {content_type} to {output_file}")
        
        elif content_type in ['images', 'videos']:
            # Create a subfolder for this extraction
            media_folder = os.path.join(output_folder, filename)
            os.makedirs(media_folder, exist_ok=True)
            
            # Download media files
            for i, item in enumerate(all_content, 1):
                extension = '.jpg' if content_type == 'images' else '.mp4'
                output_file = os.path.join(media_folder, f"{i}{extension}")
                await client.download_media(item, file=output_file)
            print(f"\nSaved {len(all_content)} {content_type} to {media_folder}")

        print(f"\nExtracted {content_type}:")
        for idx, item in enumerate(all_content[:10], start=1):
            print(f"{idx}. {item}")
        if len(all_content) > 10:
            print(f"... and {len(all_content) - 10} more.")

    except Exception as e:
        print(f"An error occurred: {e}")
        # import traceback
        # traceback.print_exc()


# Context manager for managing session
@asynccontextmanager
async def manage_session(client):
    try:
        await asyncio.to_thread(delete_session_files)
        await client.start(phone=phone_number)
        yield
    finally:
        await client.disconnect()
        await asyncio.to_thread(delete_session_files, cleanup=True)


# Function to delete session files
def delete_session_files(cleanup=False):
    for session_file in glob.glob('*.session'):
        try:
            os.remove(session_file)
            if cleanup:
                print(f"Cleaned up session file: {session_file}")
        except PermissionError:
            print(f"Permission denied when trying to delete {session_file}")


# Function to retry with exponential backoff
async def retry_with_backoff(func, max_retries=MAX_RETRIES, initial_delay=1):
    for retries in range(max_retries):
        try:
            return await func()
        except (FloodWaitError, Exception) as e:
            if isinstance(e, FloodWaitError):
                wait_time = e.seconds
            elif "A wait of" in str(e):
                wait_time = int(str(e).split()[4])
            else:
                wait_time = initial_delay * (2 ** retries)
                print(f"An error occurred: {e}")

            ready_time = datetime.now() + timedelta(seconds=wait_time)
            print(f"{'Rate limited' if isinstance(e, FloodWaitError) else 'Error'}: Ready at {ready_time.strftime('%H:%M')} (in {timedelta(seconds=wait_time)})")
            await asyncio.sleep(wait_time)
    raise Exception("Max retries reached")


# Function to list chats with unread messages
async def list_unread_chats(client):
    print("\nFetching chats with unread messages ...")
    unread_chats = []
    async for dialog in client.iter_dialogs():
        if dialog.unread_count > 0:
            unread_chats.append((dialog.name, dialog.unread_count))
    
    if unread_chats:
        print("\nChats with unread messages:")
        for name, count in sorted(unread_chats, key=lambda x: x[1], reverse=True):
            print(f"- {name}: {count} unread message(s)")
    else:
        print("No chats with unread messages found.")


# Function to search messages
async def search_messages(client):
    try:
        print("\nSearching messages ...")
        chat_name = input("Chat to search in: ").strip()
        keyword = input("Keyword to search: ").strip()
        limit = int(input("Maximum messages: "))

        dialogs = await client.get_dialogs()
        target_chat = next((dialog for dialog in dialogs if dialog.name.lower() == chat_name.lower()), None)

        if not target_chat:
            print(f"'{chat_name}' not found. Please check the name and try again.")
            return

        print(f"\nSearching for '{keyword}' in '{target_chat.name}' ...")
        messages = await client.get_messages(target_chat, search=keyword, limit=limit)

        print(f"\nFound {len(messages)} messages containing '{keyword}':")
        for msg in messages:
            sender = msg.sender.first_name if msg.sender else "Unknown"
            print(f"[{msg.date}] {sender}: {msg.text[:50]} ...")

        # Save results to file
        filename = f"search_results_{chat_name}_{keyword}.txt"
        with open(filename, "w", encoding='utf-8') as file:
            for msg in messages:
                sender = msg.sender.first_name if msg.sender else "Unknown"
                file.write(f"[{msg.date}] {sender}: {msg.text}\n\n")
        print(f"\nSearch results saved to {filename}")

    except Exception as e:
        print(f"An error occurred during search: {e}")


# Main function to run the program
async def main():
    phone_number = input("Enter your phone number (including country code): ")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary session file
        session_file = os.path.join(temp_dir, f'{phone_number}-{time.time()}')
        client = TelegramClient(session_file, api_id, api_hash)
        print("Starting Telegram session ...")
        try:
            async with manage_session(client):
                await client.start(phone=phone_number)
                
                while True:
                    print("\nOptions:")
                    print("1. Extract content from chat")
                    print("2. List chats with unread messages")
                    print("3. Search for messages")
                    print("4. Quit")
                    choice = input("Enter your choice (1/2/3/4): ").strip()
                    
                    if choice == '1':
                        await retry_with_backoff(lambda: fetch_content_from_chat(client))
                    elif choice == '2':
                        await list_unread_chats(client)
                    elif choice == '3':
                        await retry_with_backoff(lambda: search_messages(client))
                    elif choice == '4':
                        print("Exiting the program ...")
                        break
                    else:
                        print("Invalid choice. Please try again.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
