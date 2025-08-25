import smtplib, ssl
from email.message import EmailMessage

def send_email_alert(smtp_server, smtp_port, username, password, to_email, subject, body):
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = username
        msg['To'] = to_email
        msg.set_content(body)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(username, password)
            server.send_message(msg)
        return True, "Email sent"
    except Exception as e:
        return False, str(e)

def send_slack_webhook(webhook_url, text):
    try:
        import json, urllib.request
        data = json.dumps({"text": text}).encode('utf-8')
        req = urllib.request.Request(webhook_url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            _ = resp.read()
        return True, "Slack notified"
    except Exception as e:
        return False, str(e)
