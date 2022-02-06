# Federated Learning training framework for raspberry Pi
This is client side application of federated learning. 

# prerequisites - Docker and Docker compose is installed 

```
ssh pi@192.168.0.123
cd ~/Desktop
sudo date -s "2022-02-05 13:22:00"
git clone git@github.com:adityajnu14/FL-Client.git or 
git clone https://github.com/adityajnu14/FL-Client.git
cd FL-Client/
docker-compose up --build
```

Then based on the menu we can train the model. Once training is completed we need to fetch the local training matrices.

