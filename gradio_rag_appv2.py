import pandas as pd
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr

# -------------------------------------------
# Globals for storing model + data
# -------------------------------------------

#Choose model to use
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# -------------------------------------------
# Load and clean CSV
# -------------------------------------------
def load_and_prepare_csv(file_obj):
    global df, index, texts

    # Load & clean
    df = pd.read_csv(file_obj.name)
    df.dropna(inplace=True)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Store row text in global list
    texts = df.apply(lambda row: " | ".join([f"{col}: {val}" for col, val in row.items()]), axis=1).tolist()

    # Build index
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings))
    index = faiss_index

    # Get unique filter values from the dataset
    return (
        f"✅ CSV loaded successfully with {len(df)} rows.",
        gr.update(choices=sorted(df['year'].unique().tolist()), value=None),
        gr.update(choices=sorted(df['local_government_area'].dropna().unique().tolist()), value=None),
        gr.update(choices=sorted(df['postcode'].dropna().astype(str).unique().tolist()), value=None),
        gr.update(choices=sorted(df['suburb'].dropna().unique().tolist()), value=None),
        gr.update(choices=sorted(df['location_division'].dropna().unique().tolist()), value=None),
        gr.update(choices=sorted(df['property_item'].dropna().unique().tolist()), value=None),
    )


# -------------------------------------------
# Send prompt to local LLM (LM Studio)
# -------------------------------------------
def call_local_llm(prompt, api_url="http://localhost:1234/v1/chat/completions", model_name="deepseek/deepseek-r1-0528-qwen3-8b"):
    headers = {"Content-Type": "application/json"}
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(api_url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# -------------------------------------------
# RAG pipeline with context + prompt output
# -------------------------------------------
def ask_question(
    question,
    year,
    lga,
    postcode,
    suburb,
    location_div,
    property_item,
    top_k=50
):
    if df is None:
        return "❌ Please upload a CSV first.", "", ""

    try:
        # Step 1: Filter using dropdowns (pandas is faster and deterministic)
        filtered_df = df.copy()
        if year: filtered_df = filtered_df[filtered_df['year'] == int(year)]
        if lga: filtered_df = filtered_df[filtered_df['local_government_area'] == lga]
        if postcode: filtered_df = filtered_df[filtered_df['postcode'].astype(str) == str(postcode)]
        if suburb: filtered_df = filtered_df[filtered_df['suburb'] == suburb]
        if location_div: filtered_df = filtered_df[filtered_df['location_division'] == location_div]
        if property_item: filtered_df = filtered_df[filtered_df['property_item'] == property_item]

        if filtered_df.empty:
            return "⚠️ No rows match the selected filters.", "", ""

        # Step 2: Convert rows to structured text (limit to 50 rows max)
        filtered_texts = filtered_df.head(top_k).apply(
            lambda row: " | ".join([f"{col}: {val}" for col, val in row.items()]),
            axis=1
        ).tolist()
        selected_rows = "\n".join(filtered_texts)

        # Step 3: Ask the LLM to analyze only the filtered rows
        prompt = f"""
You are a helpful data analysis assistant.

Below is a list of rows from a dataset of crime/property records.

Each row includes structured data about crimes, including:
- year
- local government area
- postcode
- suburb
- location division (e.g. residential)
- property item (e.g. food, other)
- number of items stolen
- value of items stolen

### DATA:
{selected_rows}

### QUESTION:
{question}

### INSTRUCTIONS:
- ONLY use the rows above.
- If multiple rows match, summarize or calculate totals.
- Do NOT guess. If no data matches, say "No data found."

### ANSWER:
"""

        answer = call_local_llm(prompt)
        return answer.strip(), selected_rows, prompt

    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""



# -------------------------------------------
# Gradio UI layout
# -------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 Local RAG Assistant (Gemma + Gradio + Filters)")
    gr.Markdown("Upload a CSV, apply filters, ask a question, and get an answer from your local LLM.")

    csv_upload = gr.File(label="Upload CSV File", file_types=[".csv"])
    upload_status = gr.Textbox(label="Upload Status", interactive=False)

    # Filter dropdowns (to be populated)
    year_dd = gr.Dropdown(label="Year", choices=[], value=None)
    lga_dd = gr.Dropdown(label="Local Government Area", choices=[], value=None)
    postcode_dd = gr.Dropdown(label="Postcode", choices=[], value=None)
    suburb_dd = gr.Dropdown(label="Suburb", choices=[], value=None)
    location_div_dd = gr.Dropdown(label="Location Division", choices=[], value=None)
    property_item_dd = gr.Dropdown(label="Property Item", choices=[], value=None)
    
    # Question input and buttons
    question_input = gr.Textbox(lines=2, placeholder="Ask a question...", label="Your Question")
    ask_button = gr.Button("Get Answer")
    
    answer_output = gr.Textbox(label="💬 LLM's Answer")
    context_output = gr.Textbox(label="📄 Source Context", lines=6, interactive=False)
    prompt_output = gr.Textbox(label="🧪 Raw Prompt", lines=6, interactive=False)

    # CSV upload populates filters
    csv_upload.change(
        fn=load_and_prepare_csv,
        inputs=csv_upload,
        outputs=[
            upload_status,
            year_dd,
            lga_dd,
            postcode_dd,
            suburb_dd,
            location_div_dd,
            property_item_dd
        ]
    )


    # Ask button triggers query with filters
    ask_button.click(
        fn=ask_question,
        inputs=[
            question_input,
            year_dd, lga_dd, postcode_dd, suburb_dd,
            location_div_dd, property_item_dd
        ],
        outputs=[answer_output, context_output, prompt_output]
    )


# -------------------------------------------
# Launch the app
# -------------------------------------------
demo.launch()


#RAG Questions
#What is the most stolen item (Number of Items) for Year = 2025, Postcode = 3067, Suburb = Abbotsford, Location Division = Residential
#Ans: The most stolen item for Year = 2025, Postcode = 3067, Suburb = Abbotsford, Location Division = Residential is **Other**, with a total of **65** items stolen.
