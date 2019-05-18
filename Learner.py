class Learner
    Learner(data:DataBunch,
            model:Module,
            opt_func:Callable='Adam',
            loss_func:Callable=None,
            metrics:Collection[Callable]=None,
            true_wd:bool=True,
            bn_wd:bool=True,
            wd:Floats=0.01,
            train_bn:bool=True,
            path:str=None,
            model_dir:PathOrStr='models',
            callback_fns:Collection[Callable]=None,
            callbacks:Collection[Callback]=<factory>,
            layer_groups:ModuleList=None,
            add_time:bool=True,
            silent:bool=None)

class DataBunch
    DataBunch(train_dl:DataLoader,
            valid_dl:DataLoader,
            fix_dl:DataLoader=None,
            test_dl:Optional[DataLoader]=None,
            device:device=None,
            dl_tfms:Optional[Collection[Callable]]=None,
            path:PathOrStr='.',
            collate_fn:Callable='data_collate',
            no_check:bool=False)

class ItemList
    ItemList(items:Iterator[T_co],
            path:PathOrStr='.',
            label_cls:Callable=None,
            inner_df:Any=None,
            processor:Union[PreProcessor, Collection[PreProcessor]]=None,
            x:ItemList=None,
            ignore_empty:bool=False)
