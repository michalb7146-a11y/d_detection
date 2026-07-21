import os
import io
import ssl
import httplib2
ssl._create_default_https_context = ssl._create_unverified_context
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# הגדרת הרשאות גישה (Scopes)
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('/Users/deviceone/Documents/d_detection/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    # התיקון המדויק עם אותיות גדולות:
    import httplib2
    from google_auth_httplib2 import AuthorizedHttp
    
    # 1. יוצרים אובייקט תקשורת שמדלג על בדיקת ה-SSL
    http_client = httplib2.Http(disable_ssl_certificate_validation=True)
    
    # 2. מחברים את האישורים לתוך אובייקט התקשורת (שים לב ל-AuthorizedHttp עם אותיות גדולות)
    authed_http = AuthorizedHttp(creds, http=http_client)
    
    # 3. מעבירים לגוגל את ה-http המאושר
    return build('drive', 'v3', http=authed_http)


def download_file(service, file_id, file_name, local_path):
    """הורדת קובץ בודד"""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    
    print(f"מוריד קובץ: {file_name}...")
    while done is False:
        status, done = downloader.next_chunk()
    
    # שמירת התוכן לקובץ מקומי
    with open(os.path.join(local_path, file_name), 'wb') as f:
        f.write(fh.getvalue())

def download_folder(service, folder_id, local_path):
    """הורדת תיקייה בצורה רקורסיבית"""
    # יצירת התיקייה המקומית אם היא לא קיימת
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # שאילתה לקבלת כל הקבצים והתיקיות בתוך התיקייה הנוכחית
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    items = results.get('files', [])

    for item in items:
        file_id = item['id']
        file_name = item['name']
        mime_type = item['mimeType']

        # אם מדובר בתת-תיקייה, נקרא לפונקציה בצורה רקורסיבית
        if mime_type == 'application/vnd.google-apps.folder':
            sub_folder_path = os.path.join(local_path, file_name)
            download_folder(service, file_id, sub_folder_path)
        else:
            # אם זה קובץ רגיל, נוריד אותו (הערה: קבצי Google Docs/Sheets דורשים ייצוא מיוחד, קוד זה מוריד קבצים בינאריים רגילים)
            if not mime_type.startswith('application/vnd.google-apps.'):
                download_file(service, file_id, file_name, local_path)
            else:
                print(f"דילוג על קובץ מערכת של גוגל (כגון דוקס/שיטס): {file_name}")

if __name__ == '__main__':
    # התחברות לשירות
    service = get_drive_service()
    
    # מזהה התיקייה בדרייב (ניתן למצוא אותו ב-URL של התיקייה בדפדפן)
    # FOLDER_ID = "1mP6pV4lx13UXVgSh_zD8-qKajMj_9z3X" # nasa2
    # FOLDER_ID = "16yL-Uquw5oo0zuj8eI1vwqMLIMaHNHYr" # 551 device 1
    # FOLDER_ID = "1TZQOQRGElgZ_NBiDp_pbRnISkz-eAloj" # 551 device 2
    # FOLDER_ID = "1Gt5r0INiGwcIxtI3Z-TZzHhdZYjxwmHG" # AuDrok from open source data
    # FOLDER_ID = "1Q1XwJsL0up-SzBPxhfBoI4ERTTIBKEKr" # 2026.05.07_acoustics
    FOLDER_ID = "17iOFQMPrj9rOiuyyQhWjXZRUeuiLDu7l"
    
    # הנתיב המקומי במחשב שבו תרצו לשמור את התיקייה
    LOCAL_DOWNLOAD_PATH = r"/Users/deviceone/Documents/b"
    
    print("מתחיל בהורדת התיקייה...")
    download_folder(service, FOLDER_ID, LOCAL_DOWNLOAD_PATH)
    print("ההורדה הסתיימה בהצלחה!")