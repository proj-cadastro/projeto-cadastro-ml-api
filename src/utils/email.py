def extract_email_extension(email: str) -> str:
    return email.split("@")[-1] if "@" in email else ""
