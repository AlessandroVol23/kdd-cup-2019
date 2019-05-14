# dvc workflow


## Initialization

1. `pip install dvc`
2. `cd path/to/kdd-cup-2019` 
3. ` dvc init`, a .dvc/ folder is created
4. `dvc remote add sshcache ssh://ubuntu@138.246.232.191:/home/ubuntu/dvc_ssh` add remote linux virtual machine 
5. `dvc config core.remote sshcache`
6. Make changes in folder `data`
7. `dvc add data/mynewfile.xml` 
8. `pip install dvc[all]` if necessary
9. `git add data/.gitignore data/myfile.xml.dvc`
10. `dvc push` (should work if the private keys are added to the remote VM)


## Steps for the others (do once)

1. `pip install dvc`. If error, download in the [main page](https://dvc.org/)
2. `cd path/to/kdd-cup-2019` 
3. check that `.dvc/config` exists, if not see step \#4 from Initialization
4. `dvc config core.remote sshcache`
5. Make changes in folder `data`
6. `dvc add data/mynewfile.xml` and same with all changed data
7. `git add data/.gitignore data/mynewfile.xml.dvc` and same with all changed data
8. `git commit -m "#your_issue your_message"`
9. `dvc push`
10. `git push`

## Steps in normal case

1. `git pull`
2. `dvc pull`
3. Make changes in `folder`
4. `dvc add folder/mynewfile.xml` and same with all changed data
5. `git add folder/.gitignore folder/mynewfile.xml.dvc` and same with all changed data
6. `git commit -m "#your_issue message"`
7. `dvc push`
8. `git push`