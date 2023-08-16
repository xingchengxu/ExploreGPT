"""
Trains a GPT to multiply n-digit numbers.
"""

import os
import sys
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------


def get_config():

    C = CN()
    C.run_id = 0

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/multiplication'

    # data
    C.data = MultiplicationDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4  # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------


class MultiplicationDataset(Dataset):
    """
    Creates n-digit multiplication problems. For example, if n=2, then an example
    multiplication problem would be to add 85 * 50 = 4250. This problem would be
    represented as the following string for the GPT:

    "85500524"

    This is because:
    - we are discarding the * and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 4250 is encoded backwards to make the multiplication easier to learn for the
      GPT model, because of how the multiplication algorithm works.

    As one more example, the problem 6 * 39 = 234 would be encoded as:

    "06394320"

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (2 * n). When n=2, this is 8.
    At test time, we will feed in an multiplication problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next 2n digits
    correctly.
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.data_seed =1377
        C.ndigit = 5
        C.train_ndigit = 3
        C.train_ratio = 0.05
        C.test_ratio = 0.005
        C.ood_ndigit = 4
        C.ood_num_data = 1*10**3
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split  # train/test/ood_test

        # split up all multiplication problems into either training data or test data
        ndigit = self.config.ndigit
        train_ndigit = self.config.train_ndigit
        assert train_ndigit <= 4, "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (10**train_ndigit)**2  # total number of possible multiplication problems with ndigit numbers
        rng = torch.Generator()
        
        if split in ['train', 'test']:
            rng.manual_seed(self.config.data_seed)
            num_data = int((self.config.train_ratio+self.config.test_ratio)*num)
            perm = torch.randperm(num, generator=rng)[:num_data]
            num_test = int(num*self.config.test_ratio)  # 20% of the whole dataset, or only up to 500
            self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]
        else:
            rng.manual_seed(self.config.data_seed+1)
            ood_high_ndigit = self.config.ood_ndigit-train_ndigit
            num_high = 10**(ood_high_ndigit)
            perm_low = torch.randperm(num, generator=rng)[:self.config.ood_num_data]
            
            nd_low = 10**train_ndigit
            a_low = perm_low // nd_low
            b_low = perm_low % nd_low

            a_high = torch.randint(int(10**(ood_high_ndigit-1)), num_high, (self.config.ood_num_data,), generator=rng)
            b_high = torch.randint(int(10**(ood_high_ndigit-1)), num_high, (self.config.ood_num_data,), generator=rng)
            
            a = a_high*nd_low+a_low
            b = b_high*nd_low+b_low
            
            perm = a*10**self.config.ood_ndigit+b
            self.ixes = perm

    def get_vocab_size(self):
        return 10  # digits 0..9

    def get_block_size(self):
        # context window
        # a,b,a*b, n+n+2n=4n
        # but then -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        return 4*self.config.ndigit - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        train_ndigit = self.config.train_ndigit
        # given a problem index idx, first recover the associated a * b
        idx = self.ixes[idx].item()
        nd = 10**train_ndigit if self.split in ['train', 'test'] else 10**self.config.ood_ndigit
        a = idx // nd
        b = idx % nd
        # calculate the "label" of the multiplication problem a * b
        c = a * b
        # encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit*2}d' % c)[::-1]  # reverse c to make multiplication easier
        render = astr + bstr + cstr
        dix = [int(s) for s in render]  # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)  # predict the next token in the sequence
        y[:ndigit*2-1] = -1  # we will only train in the output locations. -1 will mask loss to zero
        return x, y

# -----------------------------------------------------------------------------


if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    setup_logging(config)
    set_seed(config.system.seed)

    # construct train and test datasets
    train_dataset = MultiplicationDataset(config.data, split='train')
    test_dataset = MultiplicationDataset(config.data, split='test')
    ood_test_dataset = MultiplicationDataset(config.data, split='ood_test')

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    print(config)
    # log the config
    with open(os.path.join(config.system.work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

    print(config.run_id)

    run_name = f"multiplication_run{config.run_id}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # helper function for the evaluation of a model
    def eval_split(trainer, split, max_batches=None):
        dataset = {'train': train_dataset, 'test': test_dataset, 'ood_test': ood_test_dataset}[split]
        ndigit = config.data.ndigit
        results = []
        mistakes_printed_already = 0
        factors = torch.tensor([[10**i for i in range(ndigit*2)][::-1]]).to(trainer.device)
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            # isolate the first two digits of the input sequence alone
            d1d2 = x[:, :ndigit*2]
            # let the model sample the rest of the sequence
            d1d2d3 = model.generate(d1d2, ndigit*2, do_sample=False)  # using greedy argmax, not sampling
            # isolate the last digit of the sampled sequence
            d3 = d1d2d3[:, -(ndigit*2):]
            d3 = d3.flip(1)  # reverse the digits to their "normal" order
            # decode the integers from individual digits
            d1i = (d1d2[:, :ndigit] * factors[:, -ndigit:]).sum(1)
            d2i = (d1d2[:, ndigit:ndigit*2] * factors[:, -ndigit:]).sum(1)
            d3i_pred = (d3 * factors).sum(1)
            d3i_gt = d1i * d2i  # manually calculate the ground truth
            # evaluate the correctness of the results in this batch
            correct = (d3i_pred == d3i_gt).cpu()  # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 5:  # only print up to 5 mistakes to get a sense
                    mistakes_printed_already += 1
                    print("%s GPT claims that %d * %d = %d but gt is %d" % (split, d1i[i], d2i[i], d3i_pred[i], d3i_gt[i]))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("******* %s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
        return rt.sum()

    # iteration callback
    top_score = 0

    def batch_end_callback(trainer):
        global top_score
        
        if trainer.iter_num % 100 == 0:
            model.eval()
            with torch.no_grad():
                writer.add_scalar(f"train_loss/train_loss", trainer.loss.item(), trainer.iter_num)
                writer.add_scalar(f"train_score/train_score", torch.Tensor(eval_split(trainer, 'train')), trainer.iter_num)
                writer.add_scalar(f"test_score/test_score", torch.Tensor(eval_split(trainer, 'test')), trainer.iter_num)
                writer.add_scalar(f"ood_test_score/ood_test_score", torch.Tensor(eval_split(trainer, 'ood_test')), trainer.iter_num)
            model.train()

        if trainer.iter_num % 1000 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

            # evaluate both the train and test score
            # train_max_batches = {1: None, 2: None, 3: None, 4: None, 5: None}[config.data.ndigit]  # if ndigit=2 we can afford the whole train set, ow no
            model.eval()
            with torch.no_grad():
                train_score = eval_split(trainer, 'train', max_batches=None)
                test_score = eval_split(trainer, 'test',  max_batches=None)
                ood_test_score = eval_split(trainer, 'ood_test',  max_batches=None)
            score = train_score + test_score

            # save the model if this is the best score we've seen so far
            if (score > top_score) or (trainer.iter_num % 5000 == 0):
                top_score = score
                print(f"saving model with new top score of {score}, or at iteration {trainer.iter_num}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)

            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
    writer.close()
    print("Completed!")
