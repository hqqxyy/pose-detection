function h = show_boxes(im, boxes, pa)
msize = 4;
p_no = numel(pa);

switch p_no
  case 26
    partcolor = {'g','g','y','r','r','r','r','y','y','y','m','m','m','m','y','b','b','b','b','y','y','y','c','c','c','c'};
  case 14
    partcolor = {'g','g','y','r','r','y','m','m','y','b','b','y','c','c'};
  case 18
    partcolor = {'g','g','y','r','r','r','r','y','y','y','y','b','b','b','b','y','y','y'};
  otherwise
    error('showboxes: not supported');
end
h = imshow(im); 
hold on;
drawboxes = [boxes(:, 1), boxes(:, 2),...
    boxes(:, 3) - boxes(:, 1), boxes(:, 4) - boxes(:, 2)];
for i = 1:size(drawboxes, 1)
    rectangle('Position',[drawboxes(i, 1), drawboxes(i, 2), drawboxes(i, 3), drawboxes(i, 4)], ...
        'LineWidth', 2, 'EdgeColor', partcolor{i});   
end
drawnow; hold off;
