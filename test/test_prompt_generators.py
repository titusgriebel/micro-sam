import unittest

import numpy as np
import torch

from skimage.data import binary_blobs
from skimage.measure import label
from skimage.transform import AffineTransform, warp


class TestPromptGenerators(unittest.TestCase):

    def _debug(self, mask, coordinates=None, labels=None, box=None, deformed_mask=None):
        import napari

        v = napari.Viewer()
        v.add_image(mask)

        if box is not None:
            v.add_shapes(
                [np.array(
                    [[box[0], box[1]], [box[2], box[3]]]
                )],
                shape_type="rectangle"
            )

        if coordinates is not None:
            assert labels is not None
            coordinates = np.stack(coordinates).T
            print(coordinates.shape)
            labels = labels.numpy()
            prompts = v.add_points(
                data=coordinates,
                name="prompts",
                properties={"label": labels},
                edge_color="label",
                edge_color_cycle=["#00FF00", "#FF0000"],
                symbol="o",
                face_color="transparent",
                edge_width=0.5,
                size=5,
                ndim=2
            )  # this function helps to view the (colored) background/foreground points
            prompts.edge_color_mode = "cycle"

        if deformed_mask is not None:
            v.add_labels(deformed_mask.astype("uint8"), name="deformed mask / prediction")

        napari.run()

    def _get_labels(self, n_objects):
        labels = label(binary_blobs(256))

        ids, sizes = np.unique(labels, return_counts=True)
        ids, sizes = ids[1:], sizes[1:]
        keep_ids = ids[np.argsort(sizes)[::-1][:n_objects]]

        return labels, keep_ids

    def _to_one_hot(self, labels, keep_ids):
        mask = np.zeros((len(keep_ids),) + labels.shape, dtype="float32")
        for idx, label_id in enumerate(keep_ids):
            mask[idx, labels == label_id] = 1
        return mask[:, None]

    def test_point_prompt_generator(self):
        from micro_sam.prompt_generators import PointAndBoxPromptGenerator
        from micro_sam.util import get_centers_and_bounding_boxes

        n_objects = 8
        labels, label_ids = self._get_labels(n_objects)
        centers, boxes = get_centers_and_bounding_boxes(labels)

        test_point_pairs = [(1, 0), (1, 1), (4, 3), (2, 4), (3, 9), (13, 27)]
        for (n_pos, n_neg) in test_point_pairs:
            generator = PointAndBoxPromptGenerator(n_pos, n_neg, dilation_strength=4)

            label_centers = [centers[label_id] for label_id in label_ids]
            label_boxes = [boxes[label_id] for label_id in label_ids]
            label_mask = self._to_one_hot(labels, label_ids)
            label_mask = torch.from_numpy(label_mask)

            point_coordinates, point_labels, _, _ = generator(label_mask, label_boxes, label_centers)

            n_points = n_pos + n_neg
            self.assertEqual(point_coordinates.shape, (n_objects, n_points, 2))
            self.assertEqual(point_labels.shape, (n_objects, n_points))

            for mask, coords, this_labels in zip(label_mask, point_coordinates, point_labels):
                # we need to reverse the coordinates here to match the different convention
                coords_ = (coords[:, 1].numpy(), coords[:, 0].numpy())
                expected_labels = mask[0][coords_]
                agree = (this_labels == expected_labels)

                # DEBUG: check the points in napari if they don't match
                debug = False
                if not agree.all() and debug:
                    print(n_pos, n_neg)
                    self._debug(mask, coords_, this_labels)

                self.assertTrue(agree.all())

    def test_box_prompt_generator(self):
        from micro_sam.prompt_generators import PointAndBoxPromptGenerator
        from micro_sam.util import get_centers_and_bounding_boxes

        generator = PointAndBoxPromptGenerator(0, 0, dilation_strength=0, get_point_prompts=False, get_box_prompts=True)

        n_objects = 8
        labels, label_ids = self._get_labels(n_objects)
        _, boxes = get_centers_and_bounding_boxes(labels)

        label_boxes = [boxes[label_id] for label_id in label_ids]
        label_mask = self._to_one_hot(labels, label_ids)
        label_mask = torch.from_numpy(label_mask)

        _, _, boxes, _ = generator(label_mask, label_boxes)
        self.assertTrue(boxes.shape, (n_objects, 4))

        for mask, box in zip(label_mask, boxes):
            coords = torch.where(mask[0])
            expected_box = [coo.min() for coo in coords] + [coo.max() + 1 for coo in coords]
            # convert the box back to YX axis order
            box = box.numpy()
            computed_box = [box[1], box[0], box[3], box[2]]
            self.assertEqual(expected_box, computed_box)

    def test_iterative_prompt_generator(self):
        from micro_sam.prompt_generators import IterativePromptGenerator

        def _deform_labels(labels):
            scale = np.random.uniform(low=0.9, high=1.1, size=2)
            translation = np.random.rand(2) * 5
            trafo = AffineTransform(scale=scale, translation=translation)
            deformed_labels = warp(labels, trafo.inverse, order=0, preserve_range=True).astype(labels.dtype)
            return deformed_labels

        n_tries = 5  # try five times overall to stress test this
        n_objects = 8  # use 8 objects per try
        n_points = 2  # we expect two labels for each object, one positive, one negative

        prompt_gen = IterativePromptGenerator()

        for _ in range(n_tries):
            labels, keep_ids = self._get_labels(n_objects)
            deformed_labels = _deform_labels(labels)

            label_mask = self._to_one_hot(labels, keep_ids)
            predicted_mask = self._to_one_hot(deformed_labels, keep_ids)

            prompt_mask = torch.from_numpy(label_mask).to(torch.float32)
            prompt_pred = torch.from_numpy(predicted_mask).to(torch.float32)
            point_coordinates, point_labels, _, _ = prompt_gen(prompt_mask, prompt_pred)

            self.assertEqual(point_coordinates.shape, (n_objects, n_points, 2))
            self.assertEqual(point_labels.shape, (n_objects, n_points))

            for mask, pred_mask, coords, this_labels in zip(
                label_mask, predicted_mask, point_coordinates, point_labels
            ):
                # we need to reverse the coordinates here to match the different convention
                coords_ = (coords[:, 1].numpy(), coords[:, 0].numpy())
                expected_labels = mask[0][coords_]
                agree = (this_labels.numpy() == expected_labels)

                # the label and prediction should be different for all selected points
                diff = (mask != pred_mask)[0][coords_]

                # DEBUG: check the points in napari if they don't match
                debug = False
                if not (agree.all() and diff.all()) and debug:
                    self._debug(mask, coords_, this_labels, deformed_mask=pred_mask)

                self.assertTrue(agree.all())
                self.assertTrue(diff.all())


if __name__ == "__main__":
    unittest.main()
