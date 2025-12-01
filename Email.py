

Cell 1: Configuration and Initial Setup
pythonimport pandas as pd
import numpy as np
import itertools
import networkx as nx
from datetime import timedelta
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Configuration
AMOUNT_TOLERANCE = 0.01
MAX_CLUSTER_SIZE = 5
TIME_WINDOW_DAYS = 3
PAIRWISE_NEG_SAMPLE_RATIO = 3

print("Configuration loaded successfully")
print(f"Amount Tolerance: {AMOUNT_TOLERANCE}")
print(f"Max Cluster Size: {MAX_CLUSTER_SIZE}")
print(f"Time Window: {TIME_WINDOW_DAYS} days")

Cell 2: Data Preparation and Cleaning
python# Copy the filtered dataframe
df = df_filtered.copy(deep=True)

print("Original shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# Clean empty strings
df = df.replace({"": None})

# Parse dates
df["DocumentDate"] = pd.to_datetime(df["DocumentDate"], errors="coerce", dayfirst=True)
print(f"\nNull dates: {df['DocumentDate'].isna().sum()}")

# Numeric Amount
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
print(f"\nAmount stats:\n{df['Amount'].describe()}")

# Standardize text fields
text_cols = ["DocType", "TransactionType", "BankTrfRef", "TransactionRefNo", "MerchantRefNum",
            "CR_DR", "GLRecordID", "OID", "PONumber", "CardNo", "ReceiptNumber",
            "AccountingDocNum", "AuthCode", "RefDocument", "Assignment", "StoreNumber",
            "AcquireRefNumber", "WebOrderNumber", "Source", "SourceType"]

for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": None, "None": None}).fillna("unknown")

print("\nData cleaning completed")
print(f"Final shape: {df.shape}")

Cell 3: Feature Engineering - Signed Amounts
python# Create signed amounts
df["CR_DR"] = df["CR_DR"].str.upper().fillna("unknown")
df["signed_amount"] = df["Amount"].astype(float) * df["CR_DR"].map({"CR": 1.0, "DR": -1.0}).fillna(1.0)

print("CR_DR distribution:")
print(df["CR_DR"].value_counts())
print("\nSigned amount stats:")
print(df["signed_amount"].describe())

# Check for balance (should sum to near zero if all transactions reconcile)
print(f"\nTotal signed amount sum: {df['signed_amount'].sum():.2f}")

# Display sample
print("\nSample data with signed amounts:")
print(df[["Amount", "CR_DR", "signed_amount"]].head(10))

Cell 4: Blocking Keys Creation
python# Create prefix blocking keys
df["bank_prefix_6"] = df["BankTrfRef"].apply(
    lambda x: x[:6] if isinstance(x, str) and len(x) >= 6 and x.lower() != "unknown" else None
)
df["acq_prefix_9"] = df["AcquireRefNumber"].apply(
    lambda x: x[:9] if isinstance(x, str) and len(x) >= 9 and x.lower() != "unknown" else None
)

# Combined block key
df["block_key"] = df.apply(
    lambda r: (r["bank_prefix_6"] + "|" + r["acq_prefix_9"]) 
    if (r["bank_prefix_6"] and r["acq_prefix_9"]) else None, 
    axis=1
)

print("Blocking key statistics:")
print(f"Records with bank_prefix_6: {df['bank_prefix_6'].notna().sum()}")
print(f"Records with acq_prefix_9: {df['acq_prefix_9'].notna().sum()}")
print(f"Records with valid block_key: {df['block_key'].notna().sum()}")
print(f"Records WITHOUT block_key: {df['block_key'].isna().sum()}")

print("\nBlock key distribution:")
print(df['block_key'].value_counts().head(20))

print("\nBlock size statistics:")
block_sizes = df.groupby('block_key').size()
print(block_sizes.describe())

Cell 5: Additional Feature Engineering
python# Assign unique index
df = df.reset_index(drop=True)
df["_idx"] = df.index

# Helper function for last N characters
def last_n(s, n=4):
    return s[-n:] if isinstance(s, str) and len(s) >= n else None

# Create additional features
df["card_last4"] = df["CardNo"].apply(lambda x: last_n(x, 4))
df["trxnref_last6"] = df["TransactionRefNo"].apply(lambda x: last_n(x, 6))
df["merchant_last6"] = df["MerchantRefNum"].apply(lambda x: last_n(x, 6))
df["amount_rounded"] = df["Amount"].round(2)

print("Feature engineering completed")
print("\nCard last 4 distribution:")
print(df["card_last4"].value_counts().head(10))

print("\nSample of engineered features:")
print(df[["_idx", "card_last4", "trxnref_last6", "merchant_last6", "amount_rounded"]].head(10))

Cell 6: Define Helper Functions
python# Pairwise feature function
def pairwise_features(a: pd.Series, b: pd.Series):
    feats = {}
    feats["idx_a"] = int(a["_idx"])
    feats["idx_b"] = int(b["_idx"])
    feats["abs_amount_diff"] = abs(a["Amount"] - b["Amount"])
    feats["signed_sum"] = a["signed_amount"] + b["signed_amount"]
    feats["amount_ratio"] = (a["Amount"] / b["Amount"]) if b["Amount"] != 0 else 0.0
    feats["days_diff"] = abs((a["DocumentDate"] - b["DocumentDate"]).days) if pd.notnull(a["DocumentDate"]) and pd.notnull(b["DocumentDate"]) else 9999
    feats["same_bank_prefix"] = int(bool(a["bank_prefix_6"] and b["bank_prefix_6"] and a["bank_prefix_6"] == b["bank_prefix_6"]))
    feats["same_acq_prefix"] = int(bool(a["acq_prefix_9"] and b["acq_prefix_9"] and a["acq_prefix_9"] == b["acq_prefix_9"]))
    feats["same_card_last4"] = int(bool(a["card_last4"] and b["card_last4"] and a["card_last4"] == b["card_last4"]))
    feats["exact_transref"] = int(bool(a["TransactionRefNo"] and b["TransactionRefNo"] and a["TransactionRefNo"] == b["TransactionRefNo"]))
    feats["exact_po"] = int(bool(a["PONumber"] and b["PONumber"] and a["PONumber"] == b["PONumber"]))
    feats["exact_weborder"] = int(bool(a["WebOrderNumber"] and b["WebOrderNumber"] and a["WebOrderNumber"] == b["WebOrderNumber"]))
    feats["same_store"] = int(bool(a["StoreNumber"] and b["StoreNumber"] and a["StoreNumber"] == b["StoreNumber"]))
    feats["same_doctype"] = int(a["DocType"] == b["DocType"])
    feats["same_trxtype"] = int(a["TransactionType"] == b["TransactionType"])
    return feats

# Candidate graph building
def build_candidate_graph(block_df):
    G = nx.Graph()
    for idx in block_df["_idx"]:
        G.add_node(int(idx))
    records = block_df.set_index("_idx").to_dict("index")
    idx_list = list(records.keys())
    
    edges_added = {"amount_zero": 0, "id_match": 0, "card_and_amount": 0}
    
    for i, j in itertools.combinations(idx_list, 2):
        a = records[i]; b = records[j]
        
        # Time constraint
        if pd.isnull(a["DocumentDate"]) or pd.isnull(b["DocumentDate"]):
            days_ok = True
        else:
            days_ok = abs((a["DocumentDate"] - b["DocumentDate"]).days) <= TIME_WINDOW_DAYS
        if not days_ok:
            continue

        # Amount zero-sum check
        signed_sum = a["signed_amount"] + b["signed_amount"]
        if abs(signed_sum) <= AMOUNT_TOLERANCE:
            G.add_edge(int(i), int(j), reason="amount_zero")
            edges_added["amount_zero"] += 1
            continue

        # Exact identifier matches
        strong_id_match = False
        for fld in ["TransactionRefNo", "PONumber", "WebOrderNumber", "MerchantRefNum", "AccountingDocNum", "AuthCode"]:
            if a.get(fld) and b.get(fld) and a[fld] == b[fld] and a[fld] != "unknown":
                strong_id_match = True
                break
        if strong_id_match:
            G.add_edge(int(i), int(j), reason="id_match")
            edges_added["id_match"] += 1
            continue

        # Card match + amount complement
        if a.get("card_last4") and a.get("card_last4") == b.get("card_last4"):
            if abs(a["signed_amount"] + b["signed_amount"]) <= (max(1.0, abs(a["Amount"])) * 0.05):
                G.add_edge(int(i), int(j), reason="card_and_amount")
                edges_added["card_and_amount"] += 1
                continue
    
    return G, edges_added

# Zero-sum partition finder
def find_zero_sum_partition(node_list, node_signed_amount_map, node_date_map, tol=AMOUNT_TOLERANCE, max_size=MAX_CLUSTER_SIZE):
    nodes = list(node_list)
    nodes = sorted(nodes)
    memo = {}

    def helper(remaining_tuple):
        if not remaining_tuple:
            return []
        if remaining_tuple in memo:
            return memo[remaining_tuple]
        remaining = list(remaining_tuple)
        
        for size in range(min(max_size, len(remaining)), 1, -1):
            for comb in itertools.combinations(remaining, size):
                s = sum(node_signed_amount_map[n] for n in comb)
                if abs(s) <= tol:
                    # Check time span
                    dates = [node_date_map[n] for n in comb if node_date_map[n] is not None]
                    if dates:
                        span = (max(dates) - min(dates)).days
                        if span > TIME_WINDOW_DAYS:
                            continue
                    
                    remaining_after = tuple(x for x in remaining if x not in comb)
                    rest = helper(tuple(sorted(remaining_after)))
                    if rest is not None:
                        res = [list(comb)] + rest
                        memo[remaining_tuple] = res
                        return res
        
        memo[remaining_tuple] = None
        return None

    return helper(tuple(nodes))

print("Helper functions defined successfully")

Cell 7: Test on a Single Block (Debug)
python# Pick a specific block to debug
test_block_key = df[df['block_key'].notna()]['block_key'].value_counts().index[0]
print(f"Testing with block_key: {test_block_key}")

test_block = df[df['block_key'] == test_block_key].copy()
print(f"Block size: {len(test_block)}")

print("\nTransactions in this block:")
print(test_block[['_idx', 'Amount', 'CR_DR', 'signed_amount', 'DocumentDate', 'TransactionRefNo', 'card_last4']])

# Build graph for this block
G_test, edges_test = build_candidate_graph(test_block)
print(f"\nGraph stats:")
print(f"Nodes: {G_test.number_of_nodes()}")
print(f"Edges: {G_test.number_of_edges()}")
print(f"Edge reasons: {edges_test}")

# Check connected components
components = list(nx.connected_components(G_test))
print(f"\nNumber of connected components: {len(components)}")
for i, comp in enumerate(components):
    print(f"Component {i}: {len(comp)} nodes - {sorted(list(comp))}")

Cell 8: Test Zero-Sum Partition on One Component
python# Pick first component from test block
if components:
    test_comp = list(components[0])
    print(f"Testing component with nodes: {test_comp}")
    
    # Get signed amounts
    node_signed_amount_map = {n: float(df.loc[df["_idx"]==n, "signed_amount"].iloc[0]) for n in test_comp}
    node_date_map = {n: df.loc[df["_idx"]==n, "DocumentDate"].iloc[0] for n in test_comp}
    
    print("\nNode details:")
    for n in test_comp:
        print(f"  Node {n}: signed_amount={node_signed_amount_map[n]:.2f}, date={node_date_map[n]}")
    
    print(f"\nTotal sum: {sum(node_signed_amount_map.values()):.4f}")
    
    # Attempt partition
    partition = find_zero_sum_partition(test_comp, node_signed_amount_map, node_date_map)
    
    if partition:
        print(f"\n✓ Found partition with {len(partition)} groups:")
        for i, group in enumerate(partition):
            group_sum = sum(node_signed_amount_map[n] for n in group)
            print(f"  Group {i}: nodes={group}, sum={group_sum:.4f}")
    else:
        print("\n✗ No valid partition found")

Cell 9: Run Full Clustering
pythonpredicted_clusters = {}
cluster_id_counter = 0
unresolved_components = []

# Statistics tracking
stats = {
    'total_blocks': 0,
    'blocks_with_key': 0,
    'blocks_without_key': 0,
    'total_components': 0,
    'partitioned_components': 0,
    'unresolved_components': 0,
    'total_edges': 0
}

blocks = df.groupby("block_key")
for block_key, block in blocks:
    stats['total_blocks'] += 1
    
    if block_key is None:
        stats['blocks_without_key'] += 1
        for idx in block["_idx"]:
            predicted_clusters[int(idx)] = cluster_id_counter
            cluster_id_counter += 1
        continue
    
    stats['blocks_with_key'] += 1
    block = block.sort_values("DocumentDate").reset_index(drop=True)
    G, edges_added = build_candidate_graph(block)
    stats['total_edges'] += G.number_of_edges()
    
    for comp in nx.connected_components(G):
        stats['total_components'] += 1
        comp_nodes = sorted(list(comp))
        node_signed_amount_map = {n: float(df.loc[df["_idx"]==n, "signed_amount"].iloc[0]) for n in comp_nodes}
        node_date_map = {n: df.loc[df["_idx"]==n, "DocumentDate"].iloc[0] for n in comp_nodes}
        
        partition = find_zero_sum_partition(comp_nodes, node_signed_amount_map, node_date_map)
        
        if partition is not None:
            stats['partitioned_components'] += 1
            for group in partition:
                cid = cluster_id_counter
                cluster_id_counter += 1
                for idx in group:
                    predicted_clusters[int(idx)] = cid
        else:
            stats['unresolved_components'] += 1
            # Greedy fallback...
            nodes_left = set(comp_nodes)
            groups = []
            while nodes_left:
                if len(nodes_left) == 1:
                    n = nodes_left.pop()
                    groups.append([n])
                    break
                
                best_pair = None
                best_val = float("inf")
                for a, b in itertools.combinations(nodes_left, 2):
                    val = abs(node_signed_amount_map[a] + node_signed_amount_map[b])
                    days_ok = True
                    if node_date_map[a] is not None and node_date_map[b] is not None:
                        days_ok = abs((node_date_map[a] - node_date_map[b]).days) <= TIME_WINDOW_DAYS
                    if days_ok and val < best_val:
                        best_val = val
                        best_pair = (a, b)
                
                if best_pair and best_val <= (max(abs(node_signed_amount_map[best_pair[0]]), abs(node_signed_amount_map[best_pair[1]])) * 0.2):
                    groups.append([best_pair[0], best_pair[1]])
                    nodes_left.remove(best_pair[0])
                    nodes_left.remove(best_pair[1])
                else:
                    for n in list(nodes_left):
                        groups.append([n])
                        nodes_left.remove(n)
            
            for group in groups:
                cid = cluster_id_counter
                cluster_id_counter += 1
                for idx in group:
                    predicted_clusters[int(idx)] = cid
            
            unresolved_components.append(list(comp_nodes))

df["pred_cluster"] = df["_idx"].apply(lambda x: predicted_clusters.get(int(x), -1))

print("Clustering Statistics:")
for k, v in stats.items():
    print(f"  {k}: {v}")
print(f"\nTotal clusters created: {cluster_id_counter}")

Cell 10: Analyze Clustering Results
pythonprint("Cluster size distribution:")
cluster_sizes = df.groupby('pred_cluster').size()
print(cluster_sizes.value_counts().sort_index())

print("\nLargest clusters:")
print(cluster_sizes.sort_values(ascending=False).head(10))

# Show some example clusters
print("\n=== Example Cluster 1 ===")
example_cluster_id = cluster_sizes[cluster_sizes > 1].index[0]
example_cluster = df[df['pred_cluster'] == example_cluster_id]
print(example_cluster[['_idx', 'pred_cluster', 'Amount', 'CR_DR', 'signed_amount', 'DocumentDate', 'TransactionRefNo']])
print(f"Cluster sum: {example_cluster['signed_amount'].sum():.4f}")

Cell 11: Evaluation (if MatchGroupId exists)
pythonif "MatchGroupId" in df.columns:
    # Compare with ground truth
    print("=== EVALUATION ===\n")
    
    # Cluster-level: ARI
    true_labels = df["MatchGroupId"].astype(str).tolist()
    pred_labels = df["pred_cluster"].astype(str).tolist()
    ari = adjusted_rand_score(true_labels, pred_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    
    # Show confusion examples
    print("\n=== Mismatched Examples ===")
    mismatch = df[df.groupby('MatchGroupId')['pred_cluster'].transform('nunique') > 1]
    if len(mismatch) > 0:
        print(f"Found {len(mismatch)} transactions in split ground-truth clusters")
        print(mismatch[['MatchGroupId', 'pred_cluster', 'Amount', 'signed_amount', 'TransactionRefNo']].head(20))
else:
    print("No MatchGroupId column found - skipping evaluation")