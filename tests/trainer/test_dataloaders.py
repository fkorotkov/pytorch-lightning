import platform

import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate


@pytest.mark.parametrize('train_percent_check', [-0.1, 1.1])
def test_train_dataloader_config(tmpdir, train_percent_check):

    model = EvalModelTemplate(tutils.get_default_hparams())
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        train_percent_check=train_percent_check,
    )

    with pytest.raises(ValueError):
        # fit model
        trainer.fit(model)


@pytest.mark.parametrize('val_check_interval', [10000, 1.1])
def test_val_dataloader_config(tmpdir, val_check_interval):

    model = EvalModelTemplate(tutils.get_default_hparams())
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_check_interval=val_check_interval,
    )

    with pytest.raises(ValueError):
        # fit model
        trainer.fit(model)


def test_multiple_val_dataloader(tmpdir):
    """Verify multiple val_dataloader."""

    model = EvalModelTemplate(tutils.get_default_hparams())

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=1.0,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # verify training completed
    assert result == 1

    # verify there are 2 val loaders
    assert len(trainer.val_dataloaders) == 2, \
        'Multiple val_dataloaders not initiated properly'

    # make sure predictions are good for each val set
    for dataloader in trainer.val_dataloaders:
        tutils.run_prediction(dataloader, trainer.model)


def test_multiple_test_dataloader(tmpdir):
    """Verify multiple test_dataloader."""

    model = EvalModelTemplate(tutils.get_default_hparams())
    model.test_step = model.test_step__empty
    model.test_step_end = model.test_step_end__multiple_dataloaders

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model)
    trainer.test()

    # verify there are 2 val loaders
    assert len(trainer.test_dataloaders) == 2, \
        'Multiple test_dataloaders not initiated properly'

    # make sure predictions are good for each test set
    for dataloader in trainer.test_dataloaders:
        tutils.run_prediction(dataloader, trainer.model)

    # run the test method
    trainer.test()


def test_train_dataloaders_passed_to_fit(tmpdir):
    """Verify that train dataloader can be passed to fit """

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # only train passed to fit
    model = EvalModelTemplate(tutils.get_default_hparams())
    trainer = Trainer(**trainer_options)
    fit_options = dict(train_dataloader=model._dataloader(train=True))
    result = trainer.fit(model, **fit_options)

    assert result == 1


def test_train_val_dataloaders_passed_to_fit(tmpdir):
    """ Verify that train & val dataloader can be passed to fit """

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # train, val passed to fit
    model = EvalModelTemplate(tutils.get_default_hparams())
    trainer = Trainer(**trainer_options)
    fit_options = dict(train_dataloader=model._dataloader(train=True),
                       val_dataloaders=model._dataloader(train=False))

    result = trainer.fit(model, **fit_options)
    assert result == 1
    assert len(trainer.val_dataloaders) == 1, \
        f'`val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'


def test_all_dataloaders_passed_to_fit(tmpdir):
    """Verify train, val & test dataloader can be passed to fit """

    model = EvalModelTemplate(tutils.get_default_hparams())
    model.test_step = model.test_step__empty

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # train, val and test passed to fit
    trainer = Trainer(**trainer_options)
    fit_options = dict(train_dataloader=model._dataloader(train=True),
                       val_dataloaders=model._dataloader(train=False))
    test_options = dict(test_dataloaders=model._dataloader(train=False))

    result = trainer.fit(model, **fit_options)

    trainer.test(**test_options)

    assert result == 1
    assert len(trainer.val_dataloaders) == 1, \
        f'val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'
    assert len(trainer.test_dataloaders) == 1, \
        f'test_dataloaders` not initiated properly, got {trainer.test_dataloaders}'


def test_multiple_dataloaders_passed_to_fit(tmpdir):
    """Verify that multiple val & test dataloaders can be passed to fit."""

    model = EvalModelTemplate(tutils.get_default_hparams())
    model.validation_step = model.validation_step_multiple_dataloaders
    model.test_step = model.test_step_multiple_dataloaders

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # train, multiple val and multiple test passed to fit
    trainer = Trainer(**trainer_options)
    fit_options = dict(train_dataloader=model._dataloader(train=True),
                       val_dataloaders=[model._dataloader(train=False),
                                        model._dataloader(train=False)])
    test_options = dict(test_dataloaders=[model._dataloader(train=False),
                                          model._dataloader(train=False)])

    results = trainer.fit(model, **fit_options)
    trainer.test(**test_options)

    assert len(trainer.val_dataloaders) == 2, \
        f'Multiple `val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'
    assert len(trainer.test_dataloaders) == 2, \
        f'Multiple `test_dataloaders` not initiated properly, got {trainer.test_dataloaders}'


def test_mixing_of_dataloader_options(tmpdir):
    """Verify that dataloaders can be passed to fit"""

    model = EvalModelTemplate(tutils.get_default_hparams())

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # fit model
    trainer = Trainer(**trainer_options)
    fit_options = dict(val_dataloaders=model._dataloader(train=False))
    results = trainer.fit(model, **fit_options)

    # fit model
    trainer = Trainer(**trainer_options)
    fit_options = dict(val_dataloaders=model._dataloader(train=False))
    test_options = dict(test_dataloaders=model._dataloader(train=False))

    _ = trainer.fit(model, **fit_options)
    trainer.test(**test_options)

    assert len(trainer.val_dataloaders) == 1, \
        f'`val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'
    assert len(trainer.test_dataloaders) == 1, \
        f'`test_dataloaders` not initiated properly, got {trainer.test_dataloaders}'


@pytest.mark.parametrize('check_interval', ['train', 'val', 'test'])
def test_inf_dataloader_error(tmpdir, check_interval):
    """Test inf train data loader (e.g. IterableDataset)"""

    model = EvalModelTemplate(tutils.get_default_hparams())
    if check_interval == 'train':
        model.train_dataloader = model.train_dataloader__infinite
    elif check_interval == 'val':
        model.val_dataloader = model.val_dataloader__infinite
    elif check_interval == 'test':
        model.test_dataloader = model.test_dataloader__infinite

    trainer_options = dict(default_root_dir=tmpdir, max_epochs=1)
    trainer_options[check_interval + '_check_interval'] = 0.5

    with pytest.raises(MisconfigurationException):
        trainer = Trainer(**trainer_options)
    # fit model
        trainer.fit(model)


@pytest.mark.parametrize('check_interval', [50, 1.0])
def test_inf_train_dataloader(tmpdir, check_interval):
    """Test inf train data loader (e.g. IterableDataset)"""

    model = EvalModelTemplate(tutils.get_default_hparams())
    model.train_dataloader = model.train_dataloader__infinite

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        train_check_interval=check_interval,
    )
    result = trainer.fit(model)
    # verify training completed
    assert result == 1


@pytest.mark.parametrize('check_interval', [50, 1.0])
def test_inf_val_dataloader(tmpdir, check_interval):
    """Test inf val data loader (e.g. IterableDataset)"""

    model = EvalModelTemplate(tutils.get_default_hparams())
    model.val_dataloader = model.val_dataloader__infinite

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_check_interval=check_interval,
    )
    result = trainer.fit(model)

    # verify training completed
    assert result == 1


@pytest.mark.parametrize('check_interval', [50, 1.0])
def test_inf_test_dataloader(tmpdir, check_interval):
    """Test inf test data loader (e.g. IterableDataset)"""

    model = EvalModelTemplate(tutils.get_default_hparams())
    model.test_dataloader = model.test_dataloader__infinite

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        test_check_interval=check_interval,
    )
    result = trainer.fit(model)

    # verify training completed
    assert result == 1


def test_error_on_zero_len_dataloader(tmpdir):
    """ Test that error is raised if a zero-length dataloader is defined """

    model = EvalModelTemplate(tutils.get_default_hparams())
    model.train_dataloader = model.train_dataloader__zero_length()

    # fit model
    with pytest.raises(ValueError):
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            test_percent_check=0.5
        )
        trainer.fit(model)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Does not apply to Windows platform.')
def test_warning_with_few_workers(tmpdir):
    """ Test that error is raised if dataloader with only a few workers is used """

    model = EvalModelTemplate(tutils.get_default_hparams())
    model.test_step = model.test_step__empty

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    fit_options = dict(train_dataloader=model.dataloader(train=True),
                       val_dataloaders=model.dataloader(train=False))
    test_options = dict(test_dataloaders=model.dataloader(train=False))

    trainer = Trainer(**trainer_options)

    # fit model
    with pytest.warns(UserWarning, match='train'):
        trainer.fit(model, **fit_options)

    with pytest.warns(UserWarning, match='val'):
        trainer.fit(model, **fit_options)

    with pytest.warns(UserWarning, match='test'):
        trainer.test(**test_options)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Test requires multiple GPUs')
def test_dataloader_reinit_for_subclass():

    class CustomDataLoader(torch.utils.data.DataLoader):
        def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                     batch_sampler=None, num_workers=0, collate_fn=None,
                     pin_memory=False, drop_last=False, timeout=0,
                     worker_init_fn=None, dummy_kwarg=None):
            super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                             num_workers, collate_fn, pin_memory, drop_last, timeout,
                             worker_init_fn)

            self.dummy_kwarg = dummy_kwarg

    trainer = Trainer(
        gpus=[0, 1],
        num_nodes=1,
        distributed_backend='ddp',
    )

    class CustomDummyObj:
        sampler = None

    result = trainer.auto_add_sampler(CustomDummyObj(), train=True)
    assert isinstance(result, CustomDummyObj), "Wrongly reinstantiated data loader"

    result = trainer.auto_add_sampler(CustomDataLoader(list(range(1000))), train=True)
    assert isinstance(result, torch.utils.data.DataLoader)
    assert isinstance(result, CustomDataLoader)
    assert hasattr(result, 'dummy_kwarg')
