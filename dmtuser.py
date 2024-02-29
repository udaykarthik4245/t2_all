import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
def get_items_from_user():
    items = input("Enter items purchased separated by commas: ")
    return items.split(',')
def fun2():

# Get the number of transactions from the user
    num_transactions = int(input("Enter the number of transactions: "))

    # Get transactions from user
    transactions = []
    for i in range(num_transactions):
        print(f"\nTransaction {i + 1}:")
        transaction = get_items_from_user()
        transactions.append(transaction)

    # Encode the transactions into a transaction matrix
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Find frequent item sets without setting a minimum support threshold
    frequent_itemsets = apriori(df_encoded, min_support=0.00001, use_colnames=True)

    # Print frequent item sets with headings
    for i in range(1, len(frequent_itemsets['itemsets'].apply(len).value_counts()) + 1):
        print(f"{i}-frequent items:")
        frequent_itemsets_i = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == i]
        for index, row in frequent_itemsets_i.iterrows():
            items = ', '.join(row['itemsets'])
            support = row['support']
            print(f"{index} Support: {support:.1%} Items: ({items})")
        print()
 # Find association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Determine closed item sets
    closed_itemsets = []
    for index, row in frequent_itemsets.iterrows():
        itemset = set(row['itemsets'])
        support = row['support']
        is_closed = True
        for _, rule in rules.iterrows():
            if itemset.issubset(rule['antecedents']) and support <= rule['support']:
                is_closed = False
                break
        if is_closed:
            closed_itemsets.append((itemset, support))

    # Print closed item sets
    print("Closed Item Sets:")
    for itemset, support in closed_itemsets:
        print(f"{itemset} - Support: {support:.2%}")

    # Print frequent item sets
    print("Frequent Item Sets:")
    for length in range(1, max(frequent_itemsets['itemsets'].apply(len)) + 1):
        print(f"{length}-Item Sets:")
        for index, row in frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == length].iterrows():
            items = ', '.join(row['itemsets'])
            support = row['support']
            print(f"- Items: {items}, Support: {support:.2f}")
        print()

    # Print association rules
    print("\nAssociation Rules:")
    for index, row in rules.iterrows():
        antecedents = ', '.join(row['antecedents'])
        consequents = ', '.join(row['consequents'])
        support = row['support']
        confidence = row['confidence']
        lift = row['lift']
        print(f"- Rule: {antecedents} -> {consequents}, Support: {support:.2f}, Confidence: {confidence:.2f}, Lift: {lift:.2f}")

    # Print rules with maximum support, confidence, and lift
    print("\nRules with Maximum buyers according to our analysis is :")
    max_support_rule = rules[rules['support'] == rules['support'].max()]
    if not max_support_rule.empty:
        max_support_rule = max_support_rule.iloc[0]
        print(f"- Rule with Maximum buyers: {max_support_rule['antecedents']} -> {max_support_rule['consequents']}, Support: {max_support_rule['support']:.2f}, Confidence: {max_support_rule['confidence']:.2f}, Lift: {max_support_rule['lift']:.2f}")
        print()
        print("------------------------------------------------------------------------")
        print("max_items bought are--->", max_support_rule['antecedents'], max_support_rule['consequents'])
        print("--------------------------------------------------------")

def find_max_items(df, min_support, min_confidence):
    # Convert the 'Items_list' column to a list of lists
    df['Items_list'] = df['items purchased'].str.split(',')

    # Initialize TransactionEncoder
    te = TransactionEncoder()
    # Perform one-hot encoding on the 'Items_list' column
    te_ary = te.fit(df['Items_list']).transform(df['Items_list'])
    # Convert the encoded array into a DataFrame
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Find frequent item sets
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

    # Find association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Determine closed item sets
    closed_itemsets = []
    for index, row in frequent_itemsets.iterrows():
        itemset = set(row['itemsets'])
        support = row['support']
        is_closed = True
        for _, rule in rules.iterrows():
            if itemset.issubset(rule['antecedents']) and support <= rule['support']:
                is_closed = False
                break
        if is_closed:
            closed_itemsets.append((itemset, support))

    # Print closed item sets
    print("Closed Item Sets:")
    for itemset, support in closed_itemsets:
        print(f"{itemset} - Support: {support:.2%}")

    # Print frequent item sets
    print("Frequent Item Sets:")
    for length in range(1, max(frequent_itemsets['itemsets'].apply(len)) + 1):
        print(f"{length}-Item Sets:")
        for index, row in frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == length].iterrows():
            items = ', '.join(row['itemsets'])
            support = row['support']
            print(f"- Items: {items}, Support: {support:.2f}")
        print()

    # Print association rules
    print("\nAssociation Rules:")
    for index, row in rules.iterrows():
        antecedents = ', '.join(row['antecedents'])
        consequents = ', '.join(row['consequents'])
        support = row['support']
        confidence = row['confidence']
        lift = row['lift']
        print(f"- Rule: {antecedents} -> {consequents}, Support: {support:.2f}, Confidence: {confidence:.2f}, Lift: {lift:.2f}")

    # Print rules with maximum support, confidence, and lift
    print("\nRules with Maximum buyers according to our analysis is :")
    max_support_rule = rules[rules['support'] == rules['support'].max()]
    if not max_support_rule.empty:
        max_support_rule = max_support_rule.iloc[0]
        print(f"- Rule with Maximum buyers: {max_support_rule['antecedents']} -> {max_support_rule['consequents']}, Support: {max_support_rule['support']:.2f}, Confidence: {max_support_rule['confidence']:.2f}, Lift: {max_support_rule['lift']:.2f}")
        print()
        print("------------------------------------------------------------------------")
        print("max_items bought are--->", max_support_rule['antecedents'], max_support_rule['consequents'])
        print("------------------------------------------------------------------------")


# Read the data into a pandas DataFrame
df = pd.read_csv("/content/dmt.csv")

# Get user input for minimum support and confidence
min_support = float(input("Enter the minimum support: "))
min_confidence = float(input("Enter the minimum confidence: "))

# Find frequent item sets, association rules, and maximum items
# find_max_items(df, min_support, min_confidence)
fun2()
exit()
