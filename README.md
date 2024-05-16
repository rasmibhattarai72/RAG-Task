Retrieval-Augmented Generation (RAG) System with PDF Data

This project demonstrates the implementation of a Retrieval-Augmented Generation (RAG) system. The system processes a PDF document, splits it into text chunks, embeds these chunks into vectors, stores them in a vector database, and sets up a chat model for querying the content interactively.

Installation Steps
1. Create a virtual environment.
2. Activate the virtual environment.
3. Install the required packages using the requirements.txt file.
4. Set up the environment variables in a .env file. Include your API keys for GENAI_API_KEY and GROQ_API_KEY.

Project Structure
data/: Contains the PDF file (Civil-code.pdf).
app.py: Handles document processing and model initialization.
chat_pipeline.py: Manages query handling and user interaction.
requirements.txt: Lists all required packages and dependencies.
.env: Stores sensitive data like API keys.

Environment Variables
The .env file contains sensitive data such as API keys. Keeping this information in the .env file ensures security and prevents accidental exposure of these keys in the codebase.

Usage
Running the Application
1. Ensure your virtual environment is activated.
2. Run the application using the chat_pipeline.py script.
3. Interact with the system by entering your questions when prompted. Type "exit" to close the application.
