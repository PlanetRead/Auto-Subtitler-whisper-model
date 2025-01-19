import io
import os
import pickle

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def authenticate_google_drive():
    """Authenticate with Google Drive API."""
    creds = None

    # Load existing credentials if available
    if os.path.exists("data/gdown/token.pickle"):
        with open("data/gdown/token.pickle", "rb") as token:
            creds = pickle.load(token)

    # If credentials are invalid or don't exist, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("data/gdown/credentials.json", SCOPES)
            # Try different ports if 8080 is occupied
            for port in range(8080, 8090):
                try:
                    creds = flow.run_local_server(port=port)
                    break
                except OSError:
                    continue
            else:
                raise OSError("Could not find an available port between 8080-8089")

        # Save credentials for future use
        with open("data/gdown/token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return creds


def list_folder_contents(service, folder_id):
    """List all contents of a folder for debugging."""
    try:
        results = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents",
                fields="files(id, name, mimeType)",
                pageSize=1000,
                # Add these parameters to ensure we can see all files
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )

        files = results.get("files", [])
        print(f"\nFolder contents ({len(files)} items):")
        for file in files:
            print(f"- {file['name']} ({file['mimeType']})")
        return files
    except Exception as e:
        print(f"Error listing folder contents: {str(e)}")
        return []


def download_folder(folder_id, destination_path):
    """Download all files from a Google Drive folder."""
    print(f"\nStarting download from folder ID: {folder_id}")
    creds = authenticate_google_drive()
    service = build("drive", "v3", credentials=creds)

    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"Created destination directory: {destination_path}")

    # List folder contents for debugging
    print("\nChecking folder contents...")
    files = list_folder_contents(service, folder_id)

    if not files:
        print("\nTrying alternative query method...")
        # Try alternative query method
        results = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents",
                fields="files(id, name, mimeType)",
                pageSize=1000,
                spaces="drive",
                corpora="user",
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        files = results.get("files", [])

    if not files:
        print("No files found in the specified folder. Please verify:")
        print(f"1. Folder ID: {folder_id} is correct")
        print("2. The folder is shared with your Google account")
        print("3. You have at least 'Viewer' access to the folder")
        return

    print(f"\nProcessing {len(files)} items...")

    for file in files:
        print(f"\nProcessing: {file['name']} (Type: {file['mimeType']})")
        if file["mimeType"] == "application/vnd.google-apps.folder":
            # Recursively download subfolders
            subfolder_path = os.path.join(destination_path, file["name"])
            print(f"Found subfolder: {file['name']}, downloading to {subfolder_path}")
            download_folder(file["id"], subfolder_path)
        else:
            # Download file
            try:
                request = service.files().get_media(fileId=file["id"])
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False

                while not done:
                    status, done = downloader.next_chunk()
                    print(f"Downloading {file['name']}: {int(status.progress() * 100)}%")

                fh.seek(0)
                file_path = os.path.join(destination_path, file["name"])
                with open(file_path, "wb") as f:
                    f.write(fh.read())
                    print(f"Successfully downloaded {file['name']} to {file_path}")
            except Exception as e:
                print(f"Error downloading {file['name']}: {str(e)}")


# Usage example
if __name__ == "__main__":
    # Get the folder ID from the URL
    # For example, if your URL is https://drive.google.com/drive/folders/1234567890abcdef
    # Then your folder ID is 1234567890abcdef
    FOLDER_ID = "1aLLsdOwfyLaNrhlk0BCD5yQaKOZTrdp7"  # Replace with your actual folder ID
    DESTINATION_PATH = "data/Punjabi"

    print("Starting download script...")
    print(f"Folder ID: {FOLDER_ID}")
    print(f"Destination: {DESTINATION_PATH}")
    download_folder(FOLDER_ID, DESTINATION_PATH)
    print("\nDownload script completed")
