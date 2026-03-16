import streamlit as st
import pandas as pd
import time
import io
import os
import groq
from dotenv import load_dotenv
load_dotenv()

def generate_synthetic_data_from_csv(df, num_points, user_prompt_template):
    # Use the provided system prompt
    system_prompt = (
        """
        Act as a synthetic data generator to compensate for imbalanced class. You will generate the data in the requested format for the underrepresented class the user specifies, with no preamble.
        Put your csv into a csv block. Before generating that, write 5-10 observations you have about the trends and correlations in the data (stay concise under 10 words), then generate the new data with that kind of distribution/correlation.
        You must generate exactly the amount of data points the user requests.
        """
    )
    # Convert the dataframe to CSV string (without index)
    data_csv = df.to_csv(index=False)
    # Fill the user prompt template
    user_prompt = user_prompt_template.format(num_points=num_points, data_csv=data_csv)
    try:
        synthetic_text = generate_data(system_prompt, user_prompt)
    except Exception as e:
        synthetic_text = f"[Error: {e}]"
    return synthetic_text

def generate_data(system_prompt, prompt, model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=4096):
    """Call Groq chat completion and return raw model text."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")

    client = groq.Client(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

def parse_csv_from_llm_text(text):
    """Parse CSV from an LLM response, preferring fenced csv blocks."""
    import re
    from io import StringIO
    # Try to extract a ```csv ... ``` block
    block_matches = re.findall(r"```(?:csv)?\s*\n?(.*?)```", text.strip(), flags=re.IGNORECASE | re.DOTALL)
    for block in block_matches:
        candidate = block.strip()
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        if len(lines) >= 2 and "," in lines[0]:
            csv_text = "\n".join(lines)
            return pd.read_csv(StringIO(csv_text))
    # Fallback: try to parse the whole text
    return pd.read_csv(StringIO(text.strip()))

def align_generated_csv_to_source(response_text, source_df):
    """Parse generated CSV, align schema to source_df, and return only the generated DataFrame."""
    generated_df = parse_csv_from_llm_text(response_text).copy()
    missing_columns = [col for col in source_df.columns if col not in generated_df.columns]
    extra_columns = [col for col in generated_df.columns if col not in source_df.columns]
    if missing_columns or extra_columns:
        raise ValueError(f"Schema mismatch: missing={missing_columns}, extra={extra_columns}")
    generated_df = generated_df[source_df.columns].copy()
    for col in source_df.columns:
        if pd.api.types.is_numeric_dtype(source_df[col]):
            generated_df[col] = pd.to_numeric(generated_df[col], errors="coerce")
            if pd.api.types.is_integer_dtype(source_df[col]):
                generated_df[col] = generated_df[col].round().astype("Int64")
    return generated_df

st.title("Prompt Engineering for Imbalanced Dataset")

st.write("""
Upload your CSV dataset. The app will generate synthetic data using a prompt template and the Groq LLM API, then allow you to download the result.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])



st.write("How many synthetic data points do you want to generate?")
num_points = st.number_input("Number of synthetic data points", min_value=1, max_value=100, value=10)


user_prompt_template = None
show_prompt_input = False


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Let user select the target column
    columns = df.columns.tolist()
    default_target = "DEFAULT" if "DEFAULT" in columns else columns[-1]
    target_col = st.selectbox("Select the target (predicted) column:", columns, index=columns.index(default_target))

    if st.button("Generate Prompt Template"):
        show_prompt_input = True
        # Use the actual number of points and show a sample of the uploaded data
        sample_csv = df.head(min(5, len(df))).to_csv(index=False)
        default_user_prompt = (
            f"Given this dataset, generate exactly {num_points} additional data points in CSV format, target variable is {target_col}. Maintain syntactic compatibility.\n\n{sample_csv}"
        )
        st.session_state["user_prompt_template"] = default_user_prompt

    if "user_prompt_template" in st.session_state:
        show_prompt_input = True
        user_prompt_template = st.session_state["user_prompt_template"]

    if show_prompt_input:
        user_prompt_template = st.text_area(
            "User Prompt Template (use {num_points} and {data_csv})",
            value=user_prompt_template,
            height=120,
            key="user_prompt_template_area"
        )

    generated_df = None
    progress_placeholder = st.empty()
    download_placeholder = st.empty()

    if show_prompt_input and user_prompt_template:
        if st.button("Generate Synthetic Data"):
            progress_bar = progress_placeholder.progress(0)
            # Generate synthetic data with LLM using the user-editable prompt
            synthetic_text = generate_synthetic_data_from_csv(df, num_points, user_prompt_template)
            progress_bar.progress(100)
            progress_placeholder.success("Synthetic data is ready!")

            st.write("Raw LLM output:")
            st.code(synthetic_text)

            # Try to parse and align the generated CSV to the uploaded data schema
            try:
                generated_csv_df = align_generated_csv_to_source(synthetic_text, df)
                st.write("Parsed and validated synthetic data:")
                st.dataframe(generated_csv_df.head())
                csv_bytes = generated_csv_df.to_csv(index=False).encode()
                download_placeholder.download_button(
                    label="Download Synthetic Data (validated CSV)",
                    data=csv_bytes,
                    file_name="synthetic_data.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error parsing generated CSV: {e}")
                # Try to extract and show the raw CSV block for debugging
                import re
                block_matches = re.findall(r"```(?:csv)?\s*\n?(.*?)```", synthetic_text.strip(), flags=re.IGNORECASE | re.DOTALL)
                if block_matches:
                    st.warning("Raw CSV block from LLM output (check for formatting issues):")
                    st.code(block_matches[0])
                else:
                    st.warning("No valid CSV block found in LLM output. Please check the raw output above.")
