# download cmu dataset
wget -O cmu.csv https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv

# download mturk datasets
wget -O mturk_password.csv parimarjan.github.io/keystrokes_datasets/mturk_password.csv
wget -O mturk_abc123.csv parimarjan.github.io/keystrokes_datasets/mturk_abc123.csv
wget -O mturk_123456789.csv parimarjan.github.io/keystrokes_datasets/mturk_123456789.csv
wget -O mturk_letmein.csv parimarjan.github.io/keystrokes_datasets/mturk_letmein.csv
wget -O mturk_mustang.csv parimarjan.github.io/keystrokes_datasets/mturk_mustang.csv

# download data for the indiscriminate adversary
wget -O attack_password.json parimarjan.github.io/keystrokes_datasets/attack_password.json
wget -O attack_abc123.json parimarjan.github.io/keystrokes_datasets/attack_abc123.json
wget -O attack_123456789.json parimarjan.github.io/keystrokes_datasets/attack_123456789.json
wget -O attack_letmein.json parimarjan.github.io/keystrokes_datasets/attack_letmein.json
wget -O attack_mustang.json parimarjan.github.io/keystrokes_datasets/attack_mustang.json

# download android swipes dataset
wget http://www.ms.sapientia.ro/~manyi/personality/PersonalitySwipes_full.arff

# convert to .csv format
python arff_to_csv.py
