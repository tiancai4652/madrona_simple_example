#bash
#num_port=(2 10 20 100 200 1000 2000)  #switch_num=(1 5 10 50 100 500 1000) 
# num_packet_total=1000000
#num_packet_total=10000
# num_port=(2 10 20 100 200 1000 2000)
num_port=(1) # 40 50 100 200 500 1000) #2000
# num_packet_total=20
# num_port=(2)
#switch_values=(1)
for _num_port in "${num_port[@]}"; do
    cd ./build
    make -j 60
    cd ..
    # > out.file_$_num_port #$num_packet_total #> _${num_packet_total}_${_num_port}.log
    python scripts/run.py $_num_port 
    # python scripts/run.py $_num_port  
done
