# can I the code to access the messages in Saved Messages of my Telegram and extract the links present within?

import os
import re
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from dotenv import load_dotenv

load_dotenv()

api_id = int(os.environ["TELEGRAM_API_ID"])
api_hash = os.environ["TELEGRAM_API_HASH"]
phone_number = os.environ["TELEGRAM_PHONE_NUMBER"]

# Function to extract specified content from message text
def extract_content(message):
    content = {
        "links": re.findall(r'(https?://[^\s]+)', message.text) if message.text else [],
        "text": [message.text] if message.text else [],
        "images": [f"https://t.me/c/{message.chat.id}/{message.id}"] if isinstance(message.media, MessageMediaPhoto) else [],
        "videos": [f"https://t.me/c/{message.chat.id}/{message.id}"] if isinstance(message.media, MessageMediaDocument) and message.media.document.mime_type.startswith('video') else []
    }
    return content


def validate_content_type(input_type):
    valid_types = {
        "l": "links",
        "t": "text",
        "i": "images",
        "v": "videos"
    }
    
    input_type = input_type.lower()
    return valid_types.get(input_type)


async def extract_content(message):
    content = {
        "links": re.findall(r'(https?://[^\s]+)', message.text) if message.text else [],
        "text": [message.text] if message.text else [],
        "images": [],
        "videos": []
    }

    if isinstance(message.media, MessageMediaPhoto):
        # Download the image
        path = await message.download_media(file="data/images/")
        content["images"].append(os.path.abspath(path))
    elif isinstance(message.media, MessageMediaDocument) and message.media.document.mime_type.startswith('video'):
        # Download the video
        path = await message.download_media(file="data/videos/")
        content["videos"].append(os.path.abspath(path))

    return content


# Function to fetch and extract content from specified chat
async def fetch_content_from_chat():
    session_file = 'session'
    try:
        client = TelegramClient(session_file, api_id, api_hash)
        await client.start(phone=phone_number)

        chat_name = input("\nEnter name of Chat/Group/Channel: ").strip()
        content_type = input("What to extract? ([L]inks / [T]ext / [I]mages / [V]ideos): ").strip().lower()
        content_type = validate_content_type(content_type)
        if content_type is None:
            print("Invalid content type. Please choose from L, T, I, or V.")
            return
        limit = int(input(f"How many {content_type} to extract? "))

        if not chat_name:
            target_chat = await client.get_me()
        else:
            target_chat = None
            async for dialog in client.iter_dialogs():
                if dialog.name.lower() == chat_name.lower():
                    target_chat = dialog
                    break

        if not target_chat:
            print(f"'{chat_name}' not found. Please check the name and try again.")
            return

        print(f"\nFetching messages from '{target_chat.name}'...")

        all_content = []
        message_count = 0
        async for message in client.iter_messages(target_chat):
            message_count += 1
            content = await extract_content(message)
            # if content_type in content:
            if content_type in content and content[content_type]:
                all_content.extend(content[content_type])
                print(f"Found image in message {message.id}")  # Add this line for debugging
                if len(all_content) >= limit:
                    print(f"Reached limit of {limit} images. Stopping.")  # Add this line
                    break
            if message_count % 100 == 0:
                print(f"Checked {message_count} messages... Found {len(all_content)} images so far.")

        all_content = all_content[:limit]

        print(f"\nExtracted {content_type}:")
        for idx, item in enumerate(all_content, start=1):
            print(f"{idx}. {item}")

        # Create 'data' folder if it doesn't exist
        data_folder = 'data'
        os.makedirs(data_folder, exist_ok=True)

        # Save file in the 'data' folder
        filename = f"{target_chat.name}_{content_type}.txt"
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "w", encoding='utf-8') as file:
            for item in all_content:
                file.write(item + "\n")

        print(f"\n{len(all_content)} {content_type} found and saved to '{filepath}'.")
        print(f"Checked a total of {message_count} messages.")

        # Open the folder containing the downloaded files
        if content_type in ['images', 'videos']:
            folder_path = os.path.abspath(f"data/{content_type}")
            print(f"\nOpening folder: {folder_path}")
            if sys.platform == 'win32':
                os.startfile(folder_path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', folder_path])
            else:
                subprocess.Popen(['xdg-open', folder_path])

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        await client.disconnect()
        if os.path.exists(f"{session_file}.session"):
            os.remove(f"{session_file}.session")


# Run the function to extract content
async def main():
    await fetch_content_from_chat()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
