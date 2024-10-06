import os
import re
import time
import glob
import asyncio
import tempfile
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument, Dialog
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

load_dotenv()

MAX_CHARS = 100

api_id = int(os.environ["TELEGRAM_API_ID"])
api_hash = os.environ["TELEGRAM_API_HASH"]


# Function to validate content type
def validate_content_type(input_type):
    valid_types = {
        "l": "link",
        "t": "text",
        "i": "image",
        "v": "video"
    }
    return valid_types.get(input_type.lower())


# Function to extract specified content from message text
async def extract_content(message, content_type):
    try:
        date = message.date.strftime("[%y%m%d]")
        sender = message.sender.first_name if message.sender else "Unknown"
        
        if content_type == "link" and message.text:
            links = re.findall(r'(https?://[^\s]+)', message.text)
            return [f"{date} {sender}: {link}" for link in links]
        elif content_type == "text" and message.text:
            text = message.text[:MAX_CHARS]
            return [f"{date} {sender}: {text}" if len(message.text) <= MAX_CHARS else f"{date} {sender}: {text} ..."]
        elif content_type == "image" and isinstance(message.media, MessageMediaPhoto):
            os.makedirs("data/images", exist_ok=True)
            path = await message.download_media(file="data/images/")
            file_name = os.path.basename(path)
            name, ext = os.path.splitext(file_name)
            if not ext:
                new_path = f"{path}.jpg"
                os.rename(path, new_path)
                file_name = os.path.basename(new_path)
            return [f"{date} {sender}: {file_name}"]
        elif content_type == "video" and isinstance(message.media, MessageMediaDocument) and message.media.document.mime_type.startswith('video'):
            os.makedirs("data/videos", exist_ok=True)
            path = await message.download_media(file="data/videos/")
            file_name = os.path.basename(path)
            name, ext = os.path.splitext(file_name)
            if not ext:
                new_path = f"{path}.mp4"
                os.rename(path, new_path)
                file_name = os.path.basename(new_path)
            return [f"{date} {sender}: {file_name}"]
        return []
    except AttributeError as e:
        print(f"Error extracting content: {e}")
        return []


# Function to fetch and extract content from specified chat
async def fetch_content_from_chat(client):
    try:
        chat_name = input("\nEnter name of Chat/Group/Channel: ").strip()
        content_type = validate_content_type(input("What to extract? ([L]inks / [T]ext / [I]mages / [V]ideos): ").strip().lower())
        if content_type is None:
            print("Invalid content type. Please choose from L, T, I, or V.")
            return
        limit = int(input(f"How many {content_type}s to extract? "))

        dialogs = await client.get_dialogs()
        target_chat = next((dialog for dialog in dialogs if dialog.name.lower() == chat_name.lower()), None)

        if not target_chat:
            print(f"'{chat_name}' not found. Please check the name and try again.")
            return

        print(f"\nFetching messages from '{target_chat.name}' ...")

        all_content = []
        message_count = 0
        progress_interval = max(limit // 10, 20)

        async for message in client.iter_messages(target_chat, limit=None):
            message_count += 1
            content = await extract_content(message, content_type)
            if content:
                all_content.extend(content)
                print(f"Found {content_type} in message {message.id}")
                if len(all_content) >= limit:
                    break
            if message_count % progress_interval == 0:
                print(f"Checked {message_count} messages ... Found {len(all_content)} {content_type} so far.")

        all_content = all_content[:limit]

        # Save content to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_folder = f"data/{content_type}s"
        os.makedirs(content_folder, exist_ok=True)
        filename = f"{content_folder}/{chat_name}_{content_type}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for item in all_content:
                f.write(f"{item}\n")

        print(f"\nSaved {content_type}s to {filename}")
        print(f"\nExtracted {content_type}s:")
        for item in all_content:
            print(item)

    except Exception as e:
        print(f"An error occurred: {e}")


# Context manager for managing session
@asynccontextmanager
async def manage_session(client, phone_number):
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
async def retry_with_backoff(func, max_retries=5, initial_delay=1):
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
    print("Fetching chats with unread messages ...")
    
    unread_chats = []
    try:
        dialogs = await client.get_dialogs(archived=False)
        for dialog in dialogs:
            if dialog.unread_count > 0:
                unread_chats.append((dialog.name, dialog.unread_count))

        if unread_chats:
            print("Chats with unread messages (not Archived):")
            for name, count in sorted(unread_chats, key=lambda x: x[1], reverse=True):
                print(f"{name}: {count} unread message(s)")
        else:
            print("No non-archived chats with unread messages.")

        # Optionally, you can list archived chats with unread messages
        # archived_dialogs = await client.get_dialogs(archived=True)
        # archived_unread = [(d.name, d.unread_count) for d in archived_dialogs if d.unread_count > 0]
        # if archived_unread:
        #     print("\nArchived chats with unread messages (not included in the main list):")
        #     for name, count in sorted(archived_unread, key=lambda x: x[1], reverse=True):
        #         print(f"- {name}: {count} unread message(s)")

    except Exception as e:
        print(f"Error listing unread chats: {e}")


# Function to search messages
async def search_messages(client):
    try:
        print("\nSearch messages ...")
        chat_name = input("Chat to search in: ")
        keyword = input("Keyword to search: ")
        limit = int(input("Maximum messages: "))

        dialogs = await client.get_dialogs()
        target_chat = next((dialog for dialog in dialogs if dialog.name.lower() == chat_name.lower()), None)

        if not target_chat:
            print(f"'{chat_name}' not found. Please check the name and try again.")
            return

        print(f"\nSearching for '{keyword}' in '{target_chat.name}' ...")

        messages = []
        async for message in client.iter_messages(target_chat, search=keyword, limit=limit):
            date = message.date.strftime("[%y%m%d]")
            sender = message.sender.first_name if message.sender else "Unknown"
            text = message.text[:MAX_CHARS]
            messages.append(f"{date} {sender}: {text}" if len(message.text) <= MAX_CHARS else f"{date} {sender}: {text} ...")

        if messages:
            print(f"\nFound {len(messages)} messages containing '{keyword}':")
            for msg in messages:
                print(msg)

            # Save search results to file in data/search folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            search_folder = "data/search"
            os.makedirs(search_folder, exist_ok=True)
            filename = f"{search_folder}/search_results_{chat_name}_{keyword}_{timestamp}.txt"

            with open(filename, 'w', encoding='utf-8') as f:
                for msg in messages:
                    f.write(f"{msg}\n")

            print(f"\nSearch results saved to {filename}")
        else:
            print(f"\nNo messages found containing '{keyword}'")

    except Exception as e:
        print(f"An error occurred: {e}")


# Main function to run the program
async def main():
    phone_number = input("Enter your phone number (include country code): ")
    with tempfile.TemporaryDirectory() as temp_dir:
        session_file = os.path.join(temp_dir, f'{phone_number}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.session')
        client = TelegramClient(session_file, api_id, api_hash)
        print("Starting Telegram session ...")
        try:
            async with manage_session(client, phone_number):
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
