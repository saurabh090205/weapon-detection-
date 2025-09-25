import smtplib
from email.message import EmailMessage

def send_email_alert(subject, body, to_email, from_email, from_password):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, from_password)
    server.send_message(msg)
    server.quit()

# Example usage: fill in your details below
weapon_detected = True
if weapon_detected:
    send_email_alert(
        "Weapon Detected!",
        "A weapon has been detected by surveillance.",
        "saurabh.gangurde24@vit.edu",      # Verified recipient
        "ethangade96@gmail.com",        # Sender Gmail address
        "lzoe yzqu urrl rsdk"           # 16-character Gmail App Password
    )
