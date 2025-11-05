
iterations=2000
gpu=0
data="/home/not0513/data/orinY/processed/training/20250702_133223_Q2517" # Replace it with the path where you unzipped the file
output_dir="/home/not0513/data/orinY/processed/test/20250702_133223_Q2517"
caseid="20250702_133223_Q2517"
sensorid=0
python3 train.py -s ${data} --caseid ${caseid} --gpu ${gpu} --iterations ${iterations} -m ${output_dir} --max_depth 200


