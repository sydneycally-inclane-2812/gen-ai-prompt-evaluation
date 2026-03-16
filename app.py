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
    # Fill the user prompt template (no formatting)
    user_prompt = user_prompt_template
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

def column_value_distributions(df):
    """
    Return the value distribution (normalized) for each column in the dataframe.
    For numeric columns, returns value counts of binned values (10 bins).
    For categorical/low-cardinality columns, returns value counts directly.
    Returns a dictionary: {column: value_counts}
    """
    distributions = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Bin numeric columns for easier viewing
            binned = pd.cut(df[col], bins=10) if df[col].nunique() > 10 else df[col]
            dist = binned.value_counts(normalize=True, dropna=False).sort_index()
        else:
            dist = df[col].value_counts(normalize=True, dropna=False)
        distributions[col] = dist
    return distributions

# Example usage:
# dists = column_value_distributions(df)
# for col, dist in dists.items():
#     print(f"Distribution for {col}:")
#     print(dist)

def split_by_binary_column(df, column):
    """
    Split a DataFrame into two DataFrames based on a binary column value (0/1).
    Returns (df_0, df_1) where df_0 contains rows with column==0 and df_1 with column==1.
    """
    df_0 = df[df[column] == 0].copy()
    df_1 = df[df[column] == 1].copy()
    return df_0, df_1

# Example usage:
# df_0, df_1 = split_by_binary_column(imbalanced_df, 'DEFAULT')

def parse_csv_from_llm_text(text, source_df):
    """Parse CSV from an LLM response, using source_df columns to find the correct header line and parse the CSV block from there."""
    import re
    from io import StringIO
    # Build the expected header string from source_df columns, with quotes
    expected_header = ",".join([f'"{col}"' for col in source_df.columns])
    # Try to extract a ```csv ... ``` block
    block_matches = re.findall(r"```(?:csv)?\s*\n?(.*?)```", text.strip(), flags=re.IGNORECASE | re.DOTALL)
    for block in block_matches:
        candidate = block.strip()
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        # Find the line that matches the expected header (quoted)
        for i, line in enumerate(lines):
            if line.replace(' ', '') == expected_header.replace(' ', ''):
                # Found the header, take all lines from here
                csv_text = "\n".join(lines[i:])
                try:
                    return pd.read_csv(StringIO(csv_text))
                except Exception:
                    continue
        # If not found, fallback to first line with comma
        for i, line in enumerate(lines):
            if "," in line:
                csv_text = "\n".join(lines[i:])
                try:
                    return pd.read_csv(StringIO(csv_text))
                except Exception:
                    continue
    # Fallback: try to parse the whole text, searching for the header
    all_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    for i, line in enumerate(all_lines):
        if line.replace(' ', '') == expected_header.replace(' ', ''):
            csv_text = "\n".join(all_lines[i:])
            try:
                return pd.read_csv(StringIO(csv_text))
            except Exception:
                continue
    # Last fallback: try to parse the whole text as CSV
    return pd.read_csv(StringIO(text.strip()))

def align_generated_csv_to_source(response_text, source_df):
    """Parse generated CSV, align schema to source_df, and return only the generated DataFrame."""
    generated_df = parse_csv_from_llm_text(response_text, source_df).copy()
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
        imbalanced_df_only, normal_df_only = split_by_binary_column(df, default_target)
        imbalanced_df_only_dists = column_value_distributions(imbalanced_df_only)
        normal_df_only_dists = column_value_distributions(normal_df_only)
        default_user_prompt = (
            f"""
            Given this dataset, generate exactly 10 additional data points in CSV format, 
            target variable is {default_target}. Maintain syntactic compatibility. Dataframe example:
            
            \n\n{df[:10]}

            \n\n Value distributions for {default_target}=1 (imbalanced class):\n
            {imbalanced_df_only_dists}
            \n\n Value distributions for {default_target}=0 (normal class):\n
            {normal_df_only_dists}

            \n\nAvoid making data that are too similar to existing rows and avoid making normal class data
            \n\nRemember to only generate data points for the underrepresented class ({default_target}=1) and maintain the same columns and data types as the original dataset.
            """
        )
        st.session_state["user_prompt_template"] = default_user_prompt

    if "user_prompt_template" in st.session_state:
        show_prompt_input = True
        user_prompt_template = st.session_state["user_prompt_template"]

    if show_prompt_input:
        user_prompt_template = st.text_area(
            "User Prompt Template (use {num_points} and {data_csv})",
            value=user_prompt_template,
            height=320,
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
