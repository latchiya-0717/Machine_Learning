import streamlit as st
import pandas as pd
from collections import Counter
import numpy as np  # for log2 calculations

def entropy(data, target_attribute):
  """Calculates the entropy of a dataset."""
  num_instances = len(data)
  target_counts = Counter(data[target_attribute])
  entropy = 0.0
  for count in target_counts.values():
    p = count / num_instances
    entropy -= p * np.log2(p)
  return entropy

def information_gain(data, attribute, target_attribute):
  """Calculates the information gain for a specific attribute."""
  parent_entropy = entropy(data, target_attribute)
  weighted_entropy = 0.0
  unique_values = data[attribute].unique()
  for value in unique_values:
    filtered_data = data[data[attribute] == value]
    weighted_entropy += (len(filtered_data) / len(data)) * entropy(filtered_data, target_attribute)
  return parent_entropy - weighted_entropy

def choose_best_attribute(data, attributes, target_attribute):
  """Chooses the attribute with the highest information gain."""
  information_gains = {attribute: information_gain(data, attribute, target_attribute) for attribute in attributes}
  best_attribute = max(information_gains, key=information_gains.get)
  return best_attribute

def build_decision_tree(data, attributes, target_attribute):
  """Builds a decision tree recursively."""
  if len(data.groupby(target_attribute).size()) == 1:
    return data.iloc[0][target_attribute]
  if not attributes:
    return Counter(data[target_attribute]).most_common(1)[0][0]
  best_attribute = choose_best_attribute(data, attributes, target_attribute)
  tree = {best_attribute: {}}
  for value in data[best_attribute].unique():
    filtered_data = data[data[best_attribute] == value]
    remaining_attributes = attributes.copy()
    remaining_attributes.remove(best_attribute)
    subtree = build_decision_tree(filtered_data, remaining_attributes, target_attribute)
    tree[best_attribute][value] = subtree
  return tree

def classify(tree, instance):
  """Classifies a new instance using the decision tree."""
  subtree = tree
  while True:
    attribute = next(iter(subtree))
    value = instance[attribute]
    subtree = subtree[attribute].get(value)
    if isinstance(subtree, str):
      return subtree

def main():
    """Main function to run the Streamlit app."""

    st.title("Interactive ID3 Decision Tree")

    st.write("Upload a CSV dataset with a header row.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            # Allow selection of target attribute
            target_attribute = st.selectbox("Select target attribute:", list(data.columns))

            # Show data preview (optional)
            show_preview = st.checkbox("Show data preview (first 10 rows)")
            if show_preview:
                st.subheader("Data Preview:")
                st.dataframe(data.head(10))

            attributes = list(data.columns)[:-1]  # Get all attributes except the last one

            # Build the decision tree
            tree = build_decision_tree(data.copy(), attributes, target_attribute)
            st.write("Decision Tree:", tree)

            # Add features for prediction on new instances (optional)
            if st.button("Make a Prediction"):
                # Get new instance values
                new_instance = {}
                for attribute in attributes:
                    new_instance[attribute] = st.text_input(f"Enter value for {attribute}:")
                new_instance = pd.Series(new_instance)

                # Classify the new instance
                prediction = classify(tree, new_instance)
                st.write(f"Predicted Class: {prediction}")

        except Exception as e:
            st.error("Error processing dataset: {}".format(e))

if __name__ == "__main__":
    main()
