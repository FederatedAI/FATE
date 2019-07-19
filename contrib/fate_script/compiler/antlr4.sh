echo 'start to activate antlr4'
export JAVA_HOME=/data/projects/common/jdk/jdk1.8.0_192
export PATH=$PATH:$JAVA_HOME/bin
alias antlr4='java -jar //data/projects/common/jdk/jdk1.8.0_192/lib/antlr-4.5-complete.jar'
export CLASSPATH=".:/data/projects/common/jdk/jdk1.8.0_192/lib/antlr-4.5-complete.jar:$CLASSPATH"


