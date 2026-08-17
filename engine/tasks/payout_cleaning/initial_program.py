# EVOLVE-BLOCK-START
def solve(records):
    output = []
    for row in records:
        if (
            isinstance(row, dict)
            and row.get("state") == "settled"
            and row.get("currency") == "USD"
        ):
            output.append((row["account"].strip().lower(), round(row["value"], 2)))
    return output


# EVOLVE-BLOCK-END
