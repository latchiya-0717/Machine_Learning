import streamlit as st
import pandas as pd

def candidate_elimination(data, target_attribute):
  """
  Implements the Candidate Elimination Algorithm.

  Args:
      data: A pandas DataFrame containing the training data.
      target_attribute: The name of the target attribute.

  Returns:
      A tuple containing two lists:
          - General hypotheses (G)
          - Specific hypotheses (S)
  """
  num_attributes = len(data.columns)
  G = [["?"] * num_attributes for _ in range(num_attributes)]
  S = [None] * num_attributes

  for index, row in data.iterrows():
    matched_hypothesis = False
    for i in range(num_attributes):
      if S[i] is not None and S[i][i] != row[i]:
        break
      elif G[i][i] == "?" and row[i] != "?":
        G[i][i] = row[i]

    if not matched_hypothesis:
      new_hypothesis = list(row)
      for i in range(num_attributes):
        if G[i][i] != "?":
          new_hypothesis[i] = G[i][i]
      S.append(new_hypothesis)
  for i in reversed(range(len(S))):
    for j in range(i):
      if S[i] is not None and S[j] is not None and all(S[i][k] == S[j][k] for k in range(num_attributes)):
        del S[i]
        break

  return G, S

def main():
  """
  Streamlit app for the Candidate Elimination Algorithm with data preview.
  """
  st.title("BYTES BRIGADES")
  st.title("Enhanced Candidate Elimination Algorithm")
  st.write("This interactive app demonstrates the Candidate Elimination Algorithm for concept learning with data preview and more.")
  uploaded_file = st.file_uploader("Upload CSV data:", type="csv")
  if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    show_preview = st.checkbox("Show data preview (first 10 rows)")
    if show_preview:
      st.subheader("Data Preview:")
      st.dataframe(data.head(10))
    target_attribute = st.selectbox("Select target attribute:", data.columns)
    if st.button("Run Candidate Elimination"):
      G, S = candidate_elimination(data.copy(), target_attribute)
      st.subheader("General Hypotheses (G):")
      st.dataframe(pd.DataFrame(G))
      st.subheader("Specific Hypotheses (S):")
      st.dataframe(pd.DataFrame(S))
if __name__ == "__main__":
  main()
