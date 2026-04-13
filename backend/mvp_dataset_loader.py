from datasets import load_from_disk


class MVPDatasetLoader:
    def __init__(
        self,
        dataset_path="dataset/easycall_mvp_dataset",
        split_ratio=0.7,
        seed=42
    ):
        self.dataset_path = dataset_path
        self.split_ratio = split_ratio
        self.seed = seed

        self.dataset = None
        self.train_set = None
        self.test_set = None
        self.commands = None

    def _extract_commands(self):
        """Extract unique command strings."""
        commands = list(set(self.dataset["text"]))
        commands.sort()
        return commands

    def _split(self):
        """Deterministic train/test split."""
        shuffled = self.dataset.shuffle(seed=self.seed)

        split_idx = int(len(shuffled) * self.split_ratio)

        train_set = shuffled.select(range(split_idx))
        test_set = shuffled.select(range(split_idx, len(shuffled)))

        return train_set, test_set

    def load(self):
        """Load dataset and prepare splits."""
        self.dataset = load_from_disk(self.dataset_path)

        self.commands = self._extract_commands()
        self.train_set, self.test_set = self._split()

        return {
            "dataset": self.dataset,
            "train": self.train_set,
            "test": self.test_set,
            "commands": self.commands
        }