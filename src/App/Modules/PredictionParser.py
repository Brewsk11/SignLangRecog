class PredictionParser:
    queue_length = 10
    approv_seq_length = 8

    def __init__(self):
        self._letters_queue = []
        self._prev_pred = ""

    def put_and_predict(self, letter) -> str:
        self._enqueue_letter(letter)
        res = self._predict()
        return res

    def _enqueue_letter(self, letter):
        if len(self._letters_queue) >= self.queue_length:
            self._letters_queue.pop(0)

        self._letters_queue.append(letter)

    def _predict(self) -> str:
        letter_appearances: dict = {}

        for letter in self._letters_queue:
            if letter not in letter_appearances.keys():
                letter_appearances[letter] = 0

            letter_appearances[letter] += 1

        max_letter: str = ""
        max_value: int = 0
        for key, value in letter_appearances.items():
            if value > max_value:
                max_value = value
                max_letter = key

        # If this letter is the same as the previous - return
        if max_letter == self._prev_pred:
            return ""

        # If threshold is met return the letter, if not discard the noise and return nothing
        if max_value >= self.approv_seq_length:
            self._prev_pred = max_letter
            return max_letter
        else:
            return ""
