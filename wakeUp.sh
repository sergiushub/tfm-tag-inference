#!/bin/bash

echo 'starting ssh daemon'

service ssh start

echo 'starting hdfs....'

$HADOOP_HOME/bin/hdfs namenode -format

$HADOOP_HOME/sbin/start-dfs.sh

echo 'uploading data to hdfs....'
hdfs dfs -mkdir /data
hdfs dfs -put /home/data/* /data
echo 'data uploaded'

echo 'starting spark....'


echo "All started."

if [[ $1 == "-bash" ]]; then

	/bin/bash
else
	sleep infinity
fi