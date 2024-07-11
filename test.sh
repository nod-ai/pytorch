#!/bin/bash

log_file="test.log"
output_file="zoom_unimplemented_operators.log"
error_file="zoom_test_errors.log"

# python test/test_torch.py -k TestTorchDeviceTypeZOOM --verbose &> test.log
python test/test_torch.py -k TestTorchDeviceTypePRIVATEUSEONE --verbose &> $log_file

## Find Unimplemented Operator Errors from failing tests
# Pattern to search for
pattern="Could not run 'aten::[^']*' with arguments from the 'zoom' backend"

# Extract aten operators, count frequencies, sort by frequency (descending), and save to output file
grep -oP "$pattern" "$log_file" | 
sed -n "s/.*'aten::\([^']*\)'.*/\1/p" | 
sort | 
uniq -c | 
sort -rn | 
sed 's/^ *//; s/ /\t/' > "$output_file"

# Count total matches
total_matches=$(grep -cP "$pattern" "$log_file")

# Append total matches to the output file
echo -e "\nTotal unimplemented operator failures: $total_matches" >> "$output_file"
echo "A list of unimplemented operators has been saved to $output_file"

## Find assertion errors from failing tests
# Pattern to search for
pattern="Assertion Error: (*\n)"

# Extract error messages, count frequencies, sort by frequency (descending), and save to output file
# Pattern to search for
pattern="AssertionError: (.+?)(?=\n|$)"

# Extract error messages, count frequencies, sort by frequency (descending), and save to output file
grep -oP "$pattern" "$log_file" | 
sed 's/^AssertionError: //g' |
awk '{print substr($0, 1, 100)}' |  # Limit to first 100 characters
sort | 
uniq -c | 
sort -rn | 
sed 's/^ *//; s/ /\t/' > "$error_file"

# Count total matches
total_matches=$(grep -cP "$pattern" "$log_file")

# Append total matches to the output file
echo -e "\nTotal test errors failures: $total_matches" >> "$error_file"
echo "A list of test errors has been saved to $error_file"

echo "Test logs have been saved to $log_file"