# dvc workflow


## Initialization

1. `pip install dvc`
2. `cd path/to/kdd-cup-2019` 
3. `dvc init`, a .dvc/ folder is created
4. `dvc remote add sshcache ssh://ubuntu@138.246.232.191:/home/ubuntu/dvc_ssh` add remote linux virtual machine 
5. `dvc config core.remote sshcache`


## Steps for add new data

1. `git pull`
2. `dvc pull`
3. add new data file
4. `dvc add <data>`
5. `git add <data>.dvc`
5. `git commit <data>.dvc .gitignore -m "#your_issue message"`
6. `git push origin <your_branch>`
7. `dvc push` or `dvc push -r sshcache`


## Steps for push new data

1. `git pull`
2. `dvc pull`
4.  update data file
4. `dvc commit`
5. `dvc push` or `dvc push -r sshcache`


## What to do if you get error 'failed to pull data from the cloud: not a valid private RSA key file'

You need to create a new private RSA file:

1. Open GIT Bash
2. `ssh-keygen -t rsa -b 4096 -C "username@campus.lmu.de" -m PEM` - add your username
3. Overwrite, no passphrase
4. `cat ~/.ssh./id_rsa.pub | clip` - copy the new file into the clipboard
5. Go to Gitlab Settings, SSH-Keys, add new key and paste.
6. Ask anyone from the team to add your new key to the authorized keys list in the remote linux machine
7. try doing `dvc pull` and it should work. If not, ask ane


## How to add a new key in the remote linux machine

1. `ssh ubuntu@138.246.232.191`
2. If you have an error, your key isn't in the authorized keys list, ask someone else
3. vi ~/.ssh/authorized_keys 
4. paste the key
5. esc, :x and enter
