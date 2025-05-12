import os

# Set correct directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # p1/
LETTER_DIR = os.path.join(BASE_DIR, "..", "static", "letters")  # Moves up to project root
GIF_DIR = os.path.join(BASE_DIR, "..", "static", "ISL_Gif")  # Moves up to project root

def process_text(text):
    """
    Converts text into sign language image paths (GIFs or letter images).
    """
    text = text.lower().strip()
    print(f"✅ Received text: {text}")  # Debugging print

    sign_images = []

    # Check if a GIF exists for the phrase
    gif_filename = f"{text.replace(' ', '_')}.gif"
    gif_path = os.path.join(GIF_DIR, gif_filename)

    if os.path.exists(gif_path):
        print(f"✅ Found GIF: {gif_path}")
        return [f"/static/ISL_Gif/{gif_filename}"]  # Return only GIF if available

    print(f"❌ GIF not found, breaking text into letters.")

    # Convert each letter into an image path
    for char in text:
        image_filename = f"{char}.jpg"
        image_path = os.path.join(LETTER_DIR, image_filename)

        if os.path.exists(image_path):
            print(f"✅ Found image: {image_path}")  # Debugging print
            sign_images.append(f"/static/letters/{image_filename}")  # Correct path for frontend
        else:
            print(f"❌ Missing image: {image_path}")  # Debugging print

    return sign_images

# **TEST IF `main6.py` WORKS INDEPENDENTLY**
if __name__ == "__main__":
    test_text = "hello"  # Change this to any phrase for testing
    print(process_text(test_text))  # Should print paths of GIFs or images
