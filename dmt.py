from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Read the data into a pandas DataFrame
df = pd.read_csv("/content/items.csv")

# Convert the 'Items_list' column to a list of lists
df['Items_list'] = df['items purchased'].str.split(',')

# Initialize TransactionEncoder
te = TransactionEncoder()
# Perform one-hot encoding on the 'Items_list' column
te_ary = te.fit(df['Items_list']).transform(df['Items_list'])
# Convert the encoded array into a DataFrame
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
min_support=float(input("enter the minimum support:"))
min_confidence=float(input("enter the minimum confidence: "))

# Find frequent item sets
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Print frequent item sets
print("Frequent Item Sets:")
for length in range(1, max(frequent_itemsets['itemsets'].apply(len)) + 1):
    print(f"{length}-Item Sets:")
    for index, row in frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == length].iterrows():
        items = ', '.join(row['itemsets'])
        support = row['support']
        print(f"- Items: {items}, Support: {support:.2f}")
    print()

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
max_lift = 0
max_lift_rule = None

# Print association rules
print("\nAssociation Rules:")
for index, row in rules.iterrows():
    antecedents = ', '.join(row['antecedents'])
    consequents = ', '.join(row['consequents'])
    support = row['support']
    confidence = row['confidence']
    lift = row['lift']
    print(f"- Rule: {antecedents} -> {consequents}, Support: {support:.2f}, Confidence: {confidence:.2f}, Lift: {lift:.2f}")

    # Check if current rule has maximum lift
    if lift > max_lift:
        max_lift = lift
        max_lift_rule = (antecedents, consequents, support, confidence, lift)

# Print the association rule with maximum lift
if max_lift_rule:
    antecedents, consequents, support, confidence, lift = max_lift_rule
    print("------------------------------------------------------------------------")
    print()
    print(f"- Rule with Maximum Lift: {antecedents} -> {consequents}, Support: {support:.2f}, Confidence: {confidence:.2f}, Lift: {lift:.2f}")
    print()
    print("Maximum buyers bought combination of ",antecedents+','+consequents)
    print()
    print("------------------------------------------------------------------------")
else:
    print("No association rule found.")

# Print association rules
# print("\nAssociation Rules:")
# for index, row in rules.iterrows():
#     antecedents = ', '.join(row['antecedents'])
#     consequents = ', '.join(row['consequents'])
#     support = row['support']
#     confidence = row['confidence']
#     lift = row['lift']
#     print(f"- Rule: {antecedents} -> {consequents}, Support: {support:.2f}, Confidence: {confidence:.2f}, Lift: {lift:.2f}")
