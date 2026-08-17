# EVOLVE-BLOCK-START
def solve(records):
    output = []
    for row in records:
        if isinstance(row, dict) and row.get("status") == "paid":
            output.append((row["user"].strip().lower(), round(row["amount"], 2)))
    return output


# EVOLVE-BLOCK-END
