iterations=2000
gpu=0
data="/home/not0513/data/orinY/processed/training/20250702_133223_Q2517" # Replace it with the path where you unzipped the file
output_dir="/home/not0513/data/orinY/processed/test/20250702_133223_Q2517"
caseid="20250702_133223_Q2517"
sensorid=0
python3 render.py -s ${data} -m ${output_dir} --caseid ${caseid} --iteration ${iterations} --max_depth 80 --blockinfo ${output_dir}


