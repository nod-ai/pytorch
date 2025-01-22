#!/bin/bash

log_file="test.log"
bak_file="test.log.bak"
output_file="zoom_unimplemented_operators.log"
bak_out="zoom_unimplemented_operators.log.bak"
error_file="zoom_test_errors.log"
bak_err="zoom_test_errors.log.bak"

# backup logs
[ -f $log_file ] && cp $log_file $bak_file
[ -f $output_file ] && cp $output_file $bak_out
[ -f $error_file ] && cp $error_file $bak_err

python test/test_torch.py --run-parallel 0 -k TestTorchDeviceTypePRIVATEUSEONE --verbose &> $log_file
#python test/test_ops.py -k TestCommonPRIVATEUSEONE
#python test/test_ops.py -k TestCommonPRIVATEUSEONE.test_compare_cpu --verbose &> $log_file
#python test/test_ops.py -k TestCommonPRIVATEUSEONE.test_numpy_ref --verbose &> $log_file

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

## Find errors from failing tests
# Extract error messages, count frequencies, sort by frequency (descending), and save to output file
# Pattern to search for
pattern="^.*Error: (?!test)(.+?)(?=\n|$)"

grep -oP "$pattern" "$log_file" | 
sed 's/^(.*Error): //g' |
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