
import pandas as pd

input_file = "CS170 DATA LOCATIONS - SYN1.csv"
output_file = "syn1_philippine_regions_coordinates.csv"

df = pd.read_csv(
    input_file,
    header=None,
    dtype=str,
    keep_default_na=False
)

records = []

for col in range(0, df.shape[1], 3):

    region = df.iloc[0, col].strip()

    if not region:
        continue

    for row in range(2, len(df)):

        lat = df.iloc[row, col + 1].strip()
        lon = df.iloc[row, col + 2].strip()

        # Skip empty coordinates
        if not lat and not lon:
            continue

        # Check for incomplete coordinates
        if not lat or not lon:
            raise ValueError(
                f"Incomplete coordinate: {region}, row {row + 1}"
            )

        records.append({
            "region": region,
            "latitude": float(lat),
            "longitude": float(lon)
        })

cleaned = pd.DataFrame(records)

# Validate coordinates
assert cleaned["latitude"].between(-90, 90).all()
assert cleaned["longitude"].between(-180, 180).all()
assert not cleaned.isnull().any().any()

# Export CSV
cleaned.to_csv(
    output_file,
    index=False,
    encoding="utf-8",
    lineterminator="\n"
)

print("Conversion successful!")
print("Total coordinates:", len(cleaned))
print("\nCoordinates per region:")
print(cleaned.groupby("region").size())
print("\nOutput:", output_file)
