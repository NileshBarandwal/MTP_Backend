from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import confusion_matrix

from .model import build_preprocess, get_model_keras
from .utils import ensure_dir, git_sha_or_none, set_seed, write_json


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    log_mlflow = cfg.get('log_mlflow', False)
    if log_mlflow:
        import mlflow
        mlflow.start_run()
        mlflow.log_params(
            {k: cfg[k] for k in ['seed', 'epochs', 'batch_size', 'lr', 'model_name']}
        )

    set_seed(cfg['seed'])

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    rng = np.random.default_rng(cfg['seed'])
    indices = np.arange(len(x_train))
    rng.shuffle(indices)
    val_count = int(0.1 * len(x_train))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    train_map, test_map = build_preprocess()
    batch_size = cfg['batch_size']

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(train_map).shuffle(10000, seed=cfg['seed']).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(test_map).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(test_map).batch(batch_size)

    model = get_model_keras()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    class ValAccCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs and 'val_accuracy' in logs:
                val_acc = float(logs['val_accuracy'])
                print(f"Epoch {epoch + 1}: val_acc={val_acc:.4f}")
                if log_mlflow:
                    mlflow.log_metric('val_accuracy', val_acc, step=epoch + 1)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg['epochs'],
        callbacks=[ValAccCallback()],
        verbose=0,
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    if log_mlflow:
        mlflow.log_metric('test_accuracy', float(test_acc))

    preds = model.predict(test_ds)
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds.as_numpy_iterator()])
    y_pred = np.argmax(preds, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    params = {"image_size": 28, "mean": 0.1307, "std": 0.3081}
    params_path = Path(cfg['data_root']) / 'processed' / 'params.json'
    write_json(params, params_path)

    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = Path(cfg['registry_root']) / cfg['model_name'] / timestamp
    ensure_dir(model_dir)
    model.save(model_dir / 'model.h5')
    write_json(params, model_dir / 'params.json')
    write_json({"TACC": float(test_acc)}, model_dir / 'metrics.json')
    best_val_acc = max(history.history.get('val_accuracy', [0.0]))
    write_json({"best_val_acc": float(best_val_acc), "test_acc": float(test_acc)}, model_dir / 'eval_summary.json')
    write_json({"confusion_matrix": cm.tolist()}, model_dir / 'confusion_matrix.json')
    sha = git_sha_or_none()
    if sha:
        (model_dir / 'code_sha.txt').write_text(sha)
        if log_mlflow:
            mlflow.set_tag('code_sha', sha)

    if log_mlflow:
        mlflow.log_artifact(str(model_dir / 'model.h5'))
        mlflow.log_artifact(str(model_dir / 'params.json'))
        mlflow.log_artifact(str(model_dir / 'metrics.json'))
        mlflow.log_artifact(str(model_dir / 'eval_summary.json'))
        mlflow.log_artifact(str(model_dir / 'confusion_matrix.json'))

    latest_path = Path(cfg['registry_root']) / cfg['model_name'] / 'latest.txt'
    ensure_dir(latest_path.parent)
    latest_path.write_text(timestamp)

    print(str(model_dir))

    if log_mlflow:
        mlflow.end_run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
