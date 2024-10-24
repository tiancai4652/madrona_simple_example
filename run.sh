#bash
#num_port=(2 10 20 100 200 1000 2000)  #switch_num=(1 5 10 50 100 500 1000) 
# num_packet_total=1000000
# num_packet_total=10000
# num_port=(2 10 20 100 200 1000 2000)
num_env=(1) # 40 50 100 200 500 1000) #2000
# num_packet_total=20
# num_port=(2)
#switch_values=(1)
for _num_env in "${num_env[@]}"; do
    cd ./build
    make -j 60
    cd ..
    #python scripts/run.py $_num_env -ds  #--gpu  # > out.file_$_num_port #$num_packet_total #> _${num_packet_total}_${_num_port}.log
    python scripts/run.py $_num_env --gpu
done
