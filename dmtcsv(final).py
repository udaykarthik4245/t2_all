from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

def find_association_rules(df_path):
    # Read the data into a pandas DataFrame
    df = pd.read_csv(df_path)
    # Convert the 'Items_list' column to a list of lists
    df['Items_list'] = df['items purchased'].str.split(',')

    # Initialize TransactionEncoder
    te = TransactionEncoder()
    # Perform one-hot encoding on the 'Items_list' column
    te_ary = te.fit(df['Items_list']).transform(df['Items_list'])
    # Convert the encoded array into a DataFrame
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    min_support = float(input("Enter the minimum support: "))
    
    # Find frequent item sets
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

    min_confidence = float(input("Enter the minimum confidence: "))

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    max_support_rule = None
    max_confidence_rule = None
    max_lift_rule = None

    # Create a dictionary to store association rules
    association_rules_dict = {}
    for index, row in rules.iterrows():
        antecedents = ', '.join(row['antecedents'])
        consequents = ', '.join(row['consequents'])
        support = row['support']
        confidence = row['confidence']
        lift = row['lift']
        association_rules_dict[(antecedents, consequents)] = {
            'support': support,
            'confidence': confidence,
            'lift': lift
        }

    # Print association rules
    print("\nAssociation Rules:")
    for index, row in rules.iterrows():
        antecedents = ', '.join(row['antecedents'])
        consequents = ', '.join(row['consequents'])
        support = row['support']
        confidence = row['confidence']
        lift = row['lift']
        print(f"- Rule: {antecedents} -> {consequents}, Support: {support:.2f}, Confidence: {confidence:.2f}, Lift: {lift:.2f}")

        # Check for maximum support
        if max_support_rule is None or support > max_support_rule[2]:
            max_support_rule = (antecedents, consequents, support, confidence, lift)

        # Check for maximum confidence
        if max_confidence_rule is None or confidence > max_confidence_rule[3]:
            max_confidence_rule = (antecedents, consequents, support, confidence, lift)

        # Check for maximum lift
        if max_lift_rule is None or lift > max_lift_rule[4]:
            max_lift_rule = (antecedents, consequents, support, confidence, lift)

    # Print rules with maximum support, confidence, and lift
    print("\nRules with Maximum buyers according to our analysis is :")
    if max_support_rule and max_confidence_rule and max_lift_rule:
        print(f"- Rule with Maximum buyers: {max_support_rule[0]} -> {max_support_rule[1]}, Support: {max_support_rule[2]:.2f}, Confidence: {max_support_rule[3]:.2f}, Lift: {max_support_rule[4]:.2f}")
        print()
        print("------------------------------------------------------------------------")
        print("max_items bought are--->",max_support_rule[0],max_support_rule[1])
        print("------------------------------------------------------------------------")

    return association_rules_dict

def print_rule_metrics(association_rules_dict, antecedents, consequents):
    rule_key = (antecedents, consequents)
    if rule_key in association_rules_dict:
        metrics = association_rules_dict[rule_key]
        print(f"Support: {metrics['support']:.2f}, Confidence: {metrics['confidence']:.2f}, Lift: {metrics['lift']:.2f}")
    else:
        print("Association rule not found.")

# Example usage
df_path = "/content/dmt.csv"
association_rules_dict = find_association_rules(df_path)
antecedents_input = input("Enter antecedents (comma-separated): ").strip().split(',')
consequents_input = input("Enter consequents (comma-separated): ").strip().split(',')
print_rule_metrics(association_rules_dict, ', '.join(antecedents_input), ', '.join(consequents_input))



#2nd approach


from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

def find_association_rules(df_path):
    # Read the data into a pandas DataFrame
    df = pd.read_csv(df_path)
    # Convert the 'Items_list' column to a list of lists
    df['Items_list'] = df['items purchased'].str.split(',')

    # Initialize TransactionEncoder
    te = TransactionEncoder()
    # Perform one-hot encoding on the 'Items_list' column
    te_ary = te.fit(df['Items_list']).transform(df['Items_list'])
    # Convert the encoded array into a DataFrame
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    min_support = float(input("Enter the minimum support: "))

    # Find frequent item sets
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

    min_confidence = float(input("Enter the minimum confidence: "))

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    max_support_rule = None
    max_confidence_rule = None
    max_lift_rule = None
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

    # Create a dictionary to store association rules
    association_rules_dict = {}
    for index, row in rules.iterrows():
        antecedents = ', '.join(row['antecedents'])
        consequents = ', '.join(row['consequents'])
        support = row['support']
        confidence = row['confidence']
        lift = row['lift']
        association_rules_dict[(antecedents, consequents)] = {
            'support': support,
            'confidence': confidence,
            'lift': lift
        }
    

    # Print association rules
    print("\nAssociation Rules:")
    for index, row in rules.iterrows():
        antecedents = ', '.join(row['antecedents'])
        consequents = ', '.join(row['consequents'])
        support = row['support']
        confidence = row['confidence']
        lift = row['lift']
        print(f"- Rule: {antecedents} -> {consequents}, Support: {support:.2f}, Confidence: {confidence:.2f}, Lift: {lift:.2f}")

        # Check for maximum support
        if max_support_rule is None or support > max_support_rule[2]:
            max_support_rule = (antecedents, consequents, support, confidence, lift)

        # Check for maximum confidence
        if max_confidence_rule is None or confidence > max_confidence_rule[3]:
            max_confidence_rule = (antecedents, consequents, support, confidence, lift)

        # Check for maximum lift
        if max_lift_rule is None or lift > max_lift_rule[4]:
            max_lift_rule = (antecedents, consequents, support, confidence, lift)

    # Print rules with maximum support, confidence, and lift
    print("\nRules with Maximum buyers according to our analysis is :")
    if max_support_rule and max_confidence_rule and max_lift_rule:
        print(f"- Rule with Maximum buyers: {max_support_rule[0]} -> {max_support_rule[1]}, Support: {max_support_rule[2]:.2f}, Confidence: {max_support_rule[3]:.2f}, Lift: {max_support_rule[4]:.2f}")
        print()
        print("------------------------------------------------------------------------")
        print("max_items bought are--->",max_support_rule[0],max_support_rule[1])
        print("------------------------------------------------------------------------")

    return association_rules_dict

def print_rule_metrics(association_rules_dict, antecedents, consequents):
    rule_key = (antecedents, consequents)
    if rule_key in association_rules_dict:
        metrics = association_rules_dict[rule_key]
        print(f"Support: {metrics['support']:.2f}, Confidence: {metrics['confidence']:.2f}, Lift: {metrics['lift']:.2f}")
    else:
        print("Association rule not found.")

# Example usage
df_path = "/content/dmt.csv"
association_rules_dict = find_association_rules(df_path)
antecedents_input = input("Enter antecedents (comma-separated): ").strip().split(',')
consequents_input = input("Enter consequents (comma-separated): ").strip().split(',')
print_rule_metrics(association_rules_dict, ', '.join(antecedents_input), ', '.join(consequents_input))
