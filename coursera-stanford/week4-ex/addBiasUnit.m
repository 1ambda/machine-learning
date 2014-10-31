function x = addBiasUnit(x)
  bias_unit = ones(size(x, 1), 1);
  x = [bias_unit x];
end
