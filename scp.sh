tar -cvf - src/i2i.gguf | pigz -p 128 > src/i2i.tar.gz
scp src/i2i.tar.gz starmin114@1.233.219.93:~/Projects/LGHVAC_2ndyear/src/i2i.tar.gz
pigz -p 128 -dc src/i2i.tar.gz | tar -xvf -