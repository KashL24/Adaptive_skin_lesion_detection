import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DiversityLoss(nn.Layer):
    """
    Cosine Similarity Diversity Loss

    Encourages matched positive query embeddings to be
    as different as possible.
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings):
        """
        Args:
            embeddings: Tensor [K, C]

        Returns:
            Scalar diversity loss
        """

        if embeddings is None:
            return paddle.zeros([1], dtype='float32')

        if embeddings.shape[0] <= 1:
            return paddle.zeros([1], dtype='float32')

        # Normalize embeddings
        embeddings = F.normalize(embeddings, axis=1)

        # Pairwise cosine similarity
        similarity = paddle.matmul(
            embeddings,
            embeddings,
            transpose_y=True
        )

        k = similarity.shape[0]

        # Remove diagonal (self similarity)
        mask = 1.0 - paddle.eye(k)

        similarity = similarity * mask

        # Average off-diagonal cosine similarity
        loss = similarity.sum() / (k * (k - 1))

        return loss