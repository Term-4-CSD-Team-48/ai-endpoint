git reset --hard
git pull
chmod +x serve
sudo usermod -aG www-data ubuntu
sudo chown -R ubuntu:www-data /mnt/hls
sudo ./serve