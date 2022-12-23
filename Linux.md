# Linux Commands:

- Find the ubuntu version - cmd command
  <br> wsl -l -v

- Change default user of wsl - linux
  <br> Ubuntu config --default-user root

- Change the password of the username - linux
  <br> passwd <username>

- Install from the url - linx
  <br> wget <url>

- Install from sudo - linux
  <br> sudo apt install <name>

- To open home folder - linux
  <br> cd ~

- list directories - linux
  <br> ls

- make a new directory - linux
  <br> mkdir <folder name>

- chnage the direcotry - linux
  <br> cd <folder-name>/

- Rename the directory - linux
  <br> mv <curren dir name> <new dir name>

- make the folder executable - linux
  <br> chmod +x <dir name>

- make executable run from anywhere - linux
  <br> (a) come to home - cd ..
  <br> (b) nano .bashrc - add below line - export PATH ="${HOME}/installed:${PATH}"
  <br> (c) source .bashrc - running the script
  <br> (d) which <exe name> - gives the path

- To save changes to nano editor - linux
  <br> control + o

- To quit nano editor - linux 
  <br> control + x

- shutdown wsl - cmd
  <br> (a) wsl -l -v
  <br> (b) wsl --shutdown

- add tool (like docker) to user group - linux
   <br> (a) sudo groupadd docker
   <br> (b) sudo usermod -aG docker $USER

- get more details what slast software did - linux
  <br> less .batchrc

- To start a jupyter notebook (if anaconda is there) - linux
  <br>  jupyter notebook

- To get details of folder is ls - linux
  <br> ls -lh


- Cat(concatenate) command reads data from the file and gives their content as output - Linux
  <br> cat <file name>

- Get the directory name of a file 
  <br> realpath filename

- Copy a file to another folder 
  <br> cp my_file.txt /new_directory
  
- rename a file in the same directory
  <br> mv oldfilename newfilename 

- Git command to commit from wsl
  <br> git config --global user.name "Your Name"
  <br> git config --global user.email "youremail@domain.com"
  <br> If GIT installed is >= v2.36.1
  <br>  git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/bin/git-credential-manager-core.exe"
  <br> Else
  <br>  git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/libexec/git-core/git-credential-manager.exe"
