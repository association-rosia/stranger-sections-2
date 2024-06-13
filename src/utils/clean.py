import wandb
from src.utils import func
from tqdm.auto import tqdm


def main():
    api = wandb.Api()
    config = func.load_config('main', loading='object')
    runs = api.runs(f'{config.wandb.entity}/{config.wandb.project}')
    runs = [{'name': run.name, 'id': run.id} for run in runs]

    for run in tqdm(runs):
        run_data = api.run(f'{config.wandb.entity}/{config.wandb.project}/{run["id"]}')
        history = run_data.history(keys=['val/iou-micro', 'val/iou-macro'])
        max_iou_micro = max(list(history['val/iou-micro']))
        max_iou_macro = max(list(history['val/iou-macro']))
        run.update({'max_iou_micro': max_iou_micro})
        run.update({'max_iou_macro': max_iou_macro})

    print()


if __name__ == '__main__':
    main()
