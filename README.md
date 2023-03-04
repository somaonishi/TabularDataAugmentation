## Tabular Data Augmentation
## Environment
- python 3.8.10
- wsl

## Build environment
```bash
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Run command
```bash
# DA is applied as per augmenters
python main.py

# All combinations of augmenters are executed
python main_all.py
```

## Change training mode
Each config file exists directly under the directory `conf`. The mode can be switched by using the following options.
```bash
python main.py --config-name CONFIG_NAME
````
If you want to learn with `semi`, use
````bash
python main.py --config-name semi
````

## Config.yaml
You can switch the algorithm for data augmentation by modifying the following.
```yaml
augmenters: [random_flip,
             noise, 
             random_collapse, 
             shuffle, 
             random_resize
             ]
````
For example, if you want to apply only `noise` data augmentation, use the following.
```yaml
augmenters: [noise].
```
 
