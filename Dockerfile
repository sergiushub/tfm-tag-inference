FROM gcr.io/tensorflow/tensorflow:latest

MAINTAINER Sergio Dominguez Fernandez <sergio.dmgz@gmail.com>

RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y curl 

### JAVA         
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y  software-properties-common && \
    add-apt-repository ppa:webupd8team/java -y && \
    apt-get update && \
    echo oracle-java7-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections && \
    apt-get install -y oracle-java8-installer && \
    apt-get clean

# Define commonly used JAVA_HOME variable
ENV JAVA_HOME /usr/lib/jvm/java-8-oracle

### setup ssh client keys for root
RUN apt-get update && apt-get install -y openssh-client openssh-server && \
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    chmod 0600 ~/.ssh/authorized_keys && \
    ssh-keyscan -H localhost > ~/.ssh/known_hosts

ADD ssh_config /root/.ssh/config
RUN chmod 600 /root/.ssh/config && chown root:root /root/.ssh/config

### HADOOP
RUN cd /tmp && curl -L -O -k "http://www.eu.apache.org/dist/hadoop/common/hadoop-2.7.5/hadoop-2.7.5.tar.gz" && \
    tar -xf hadoop-2.7.5.tar.gz -C /opt && rm -f hadoop-2.7.5.tar.gz && \
    ln -s /opt/hadoop* /opt/hadoop

ENV HADOOP_HOME /opt/hadoop

### SPARK
RUN cd /tmp && curl -L -O -k "http://mirror.intergrid.com.au/apache/spark/spark-2.2.1/spark-2.2.1-bin-hadoop2.7.tgz" && \
    tar -xf spark-2.2.1-bin-hadoop2.7.tgz -C /opt && rm -f spark-2.2.1-bin-hadoop2.7.tgz && \
    ln -s /opt/spark* /opt/spark
ENV SPARK_HOME /opt/spark
ENV PYSPARK_DRIVER_PYTHON jupyter
ENV PYSPARK_DRIVER_PYTHON_OPTS 'notebook --allow-root'

# Add JDK,HADOOP and SPARK on PATH variable
ENV PATH ${PATH}:${JAVA_HOME}/bin:${HADOOP_HOME}/bin:${SPARK_HOME}/bin

VOLUME ["/hdfs","/var/logs"]

### Clone project 
# RUN git clone https://github.com/tensorflow/models.git
# COPY tensorflow/* /home/tensorflow/

### Copy Data into docker 
COPY data/mirflickr_test/* /home/data/

### copy hadoop config
COPY hadoop-conf/* /opt/hadoop/etc/hadoop/

### copy wakeUp.sh
COPY wakeUp.sh /home
RUN chmod +x /home/wakeUp.sh 

