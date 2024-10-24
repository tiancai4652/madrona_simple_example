if [ "$#" -ne 2 ]; then
    echo "Error: the number of parameter is $#, not 2. Please pass exactly two parameters: branch_name and remark"
    echo "Usage: \$0 <branch_name> <remark>"
    exit 1
fi

branch_name=$1
remark=$2


echo "branch_name: ${branch_name}"
echo "remark: ${remark}"


cd /data3/guifei/projects/goats/madrona_simple_example/
cp -r ./src ./scripts run.sh  /data3/guifei/projects/goats/GAND4LLM/
cp -r ./gen_fib_table  /data3/guifei/projects/goats/GAND4LLM/
cd /data3/guifei/projects/goats/GAND4LLM/
git checkout "${branch_name}"
git add .
git commit -m "${remark}"
git push -u origin "${branch_name}"
