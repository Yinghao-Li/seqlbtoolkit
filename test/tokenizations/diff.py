"""
# Author: Yinghao Li
# Modified: November 29th, 2023
# ---------------------------------------
# Description: Find sequence alignment
# Reference: https://github.com/tamuhey/seqdiff
"""


class Difference:
    def __init__(self, sequence_x, sequence_y):
        self.sequence_x = sequence_x
        self.sequence_y = sequence_y
        max_distance = len(sequence_x) + len(sequence_y) + 1
        self.offset_d = len(sequence_y)
        self.forward_vector = [float("-inf")] * max_distance
        self.backward_vector = [float("inf")] * max_distance
        self.x_endpoints = [None] * len(sequence_x)
        self.y_endpoints = [None] * len(sequence_y)

    def diff(self) -> int:
        return self.diff_part((0, len(self.sequence_x)), (0, len(self.sequence_y)))

    def diff_part(self, x_bounds, y_bounds) -> int:
        x_left, x_right = x_bounds
        y_left, y_right = y_bounds

        # Shrink by equality
        while x_left < x_right and y_left < y_right and self.sequence_x[x_left] == self.sequence_y[y_left]:
            self.x_endpoints[x_left] = y_left
            self.y_endpoints[y_left] = x_left
            x_left += 1
            y_left += 1

        while x_left < x_right and y_left < y_right and self.sequence_x[x_right - 1] == self.sequence_y[y_right - 1]:
            x_right -= 1
            y_right -= 1
            self.x_endpoints[x_right] = y_right
            self.y_endpoints[y_right] = x_right

        if x_left == x_right:
            for i in range(y_left, y_right):
                self.y_endpoints[i] = None
            return y_right - y_left
        elif y_left == y_right:
            for i in range(x_left, x_right):
                self.x_endpoints[i] = None
            return x_right - x_left
        else:
            distance, (mid_x, mid_y) = self.find_mid((x_left, x_right), (y_left, y_right))
            self.diff_part((x_left, mid_x), (y_left, mid_y))
            self.diff_part((mid_x, x_right), (mid_y, y_right))
            return distance

    def find_mid(self, x_bounds, y_bounds):
        x_left, x_right = x_bounds
        y_left, y_right = y_bounds

        diagonal_min = x_left - y_right
        diagonal_max = x_right - y_left
        forward_center_diag = x_left - y_left
        backward_center_diag = x_right - y_right
        delta = (x_right - x_left) - (y_right - y_left)
        is_odd = delta & 1 == 1

        self.forward_vector[self.ktoi(forward_center_diag)] = x_left
        self.backward_vector[self.ktoi(backward_center_diag)] = x_right

        forward_min_diag = forward_center_diag
        forward_max_diag = forward_center_diag
        backward_min_diag = backward_center_diag
        backward_max_diag = backward_center_diag

        for d in range(1, 1000):  # Assuming a large number for the upper limit
            # Forward pass
            forward_min_diag, forward_max_diag = self._update_range(
                forward_min_diag, forward_max_diag, diagonal_min, diagonal_max, self.forward_vector, float("-inf")
            )
            for k in reversed(range(forward_min_diag, forward_max_diag + 1, 2)):
                x, y = self._forward_snake(
                    k,
                    x_left,
                    x_right,
                    y_left,
                    y_right,
                    is_odd,
                    backward_min_diag,
                    backward_max_diag,
                )
                if x is not None:
                    return 2 * d - 1, (x, y)

            # Backward pass
            backward_min_diag, backward_max_diag = self._update_range(
                backward_min_diag, backward_max_diag, diagonal_min, diagonal_max, self.backward_vector, float("inf")
            )
            for k in reversed(range(backward_min_diag, backward_max_diag + 1, 2)):
                x, y = self._backward_snake(
                    k, x_left, x_right, y_left, y_right, is_odd, forward_min_diag, forward_max_diag
                )
                if x is not None:
                    return 2 * d, (x, y)

        raise Exception("Unreachable code reached")

    def _update_range(
        self,
        current_min_diag,
        current_max_diag,
        min_diag,
        max_diag,
        v,
        default_value,
    ):
        if current_min_diag > min_diag:
            current_min_diag -= 1
            v[self.ktoi(current_min_diag - 1)] = default_value
        else:
            current_min_diag += 1
        if current_max_diag < max_diag:
            current_max_diag += 1
            v[self.ktoi(current_max_diag + 1)] = default_value
        else:
            current_max_diag -= 1
        return current_min_diag, current_max_diag

    def _forward_snake(
        self,
        k,
        x_left,
        x_right,
        y_left,
        y_right,
        is_odd,
        min_backward_diag,
        max_backward_diag,
    ):
        ik = self.ktoi(k)
        x = max(
            self.forward_vector[self.ktoi(k - 1)] + 1
            if self.forward_vector[self.ktoi(k - 1)] != float("-inf")
            else float("-inf"),
            self.forward_vector[self.ktoi(k + 1)]
            if self.forward_vector[self.ktoi(k + 1)] != float("-inf")
            else float("-inf"),
        )
        y = self.get_y_from_x_and_k(x, k)
        if x_left <= x <= x_right and y_left <= y <= y_right:
            new_x, new_y = x, y
            while new_x < x_right and new_y < y_right and self.sequence_x[new_x] == self.sequence_y[new_y]:
                new_x += 1
                new_y += 1

            self.forward_vector[ik] = new_x
            if is_odd and min_backward_diag <= k <= max_backward_diag and self.backward_vector[ik] <= new_x:
                return new_x, new_y
        return None, None

    def _backward_snake(
        self,
        k,
        x_left,
        x_right,
        y_left,
        y_right,
        is_odd,
        min_forward_diag,
        max_forward_diag,
    ):
        x = min(
            self.backward_vector[self.ktoi(k - 1)]
            if self.backward_vector[self.ktoi(k - 1)] != float("inf")
            else float("inf"),
            self.backward_vector[self.ktoi(k + 1)] - 1
            if self.backward_vector[self.ktoi(k + 1)] != float("inf")
            else float("inf"),
        )
        y = self.get_y_from_x_and_k(x, k)
        if x_left <= x <= x_right and y_left <= y <= y_right:
            new_x, new_y = x, y
            while x_left < new_x and y_left < new_y and self.sequence_x[new_x - 1] == self.sequence_y[new_y - 1]:
                new_x -= 1
                new_y -= 1

            self.backward_vector[self.ktoi(k)] = new_x
            if not is_odd and min_forward_diag <= k <= max_forward_diag and self.forward_vector[self.ktoi(k)] >= new_x:
                return new_x, new_y
        return None, None

    def ktoi(self, k):
        return k + self.offset_d

    @staticmethod
    def get_y_from_x_and_k(x, k):
        return x - k


# The diff function
def diff(sequence_a, sequence_b):
    difference_tracker = Difference(sequence_a, sequence_b)
    difference_tracker.diff()
    return difference_tracker.x_endpoints, difference_tracker.y_endpoints


# The ratio function
def ratio(sequence_a, sequence_b) -> float:
    total_length = len(sequence_a) + len(sequence_b)
    if total_length == 0:
        return 100.0
    distance = Difference(sequence_a, sequence_b).diff()
    match_count = total_length - distance
    return (match_count * 100) / total_length
