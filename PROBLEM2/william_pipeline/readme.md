
### Running the system
1. python -m venv venv
2. Activate venv
3. python -m pip install -r requirements.txt
4. python final_system.py
5. put files into /videos folder
#### Cuda and MacOS silicon
The program should detect cuda and apple silicon. If not, make sure the correct pip drivers are installed. See final_system.py for more details

# Testing
## Check generated receipts against the correct ones for each video.

python evaluate_receipts.py --test "output" --reference "correct_receipts"

See the script to adjust time-window and other settings.