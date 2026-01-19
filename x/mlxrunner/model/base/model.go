package base

import (
	"encoding/json"
	"errors"
	"log/slog"
	"reflect"
	"strconv"
	"strings"

	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type Model interface {
	// Forward performs a forward pass through the model.
	Forward(inputs *mlx.Tensor, cache []cache.Cache) *mlx.Tensor

	// NumLayers returns the number of layers in the model.
	// This is used to initialize caches.
	// TODO: consider moving cache initialization into the model itself.
	NumLayers() int
}

type TextGeneration interface {
	Model
	Unembed(*mlx.Tensor) *mlx.Tensor
}

func Weights(m Model) (map[string][]*mlx.Tensor, []func() error) {
	mapping := make(map[string][]*mlx.Tensor)
	var afterLoadFuncs []func() error
	var fn func(v reflect.Value, tags []string)
	fn = func(v reflect.Value, tags []string) {
		t := v.Type()
		if t.Kind() == reflect.Pointer {
			t, v = t.Elem(), v.Elem()
		}

		if method := v.Addr().MethodByName("AfterLoad"); method.IsValid() {
			var afterLoadFunc func() error
			reflect.ValueOf(&afterLoadFunc).Elem().Set(method)
			afterLoadFuncs = append(afterLoadFuncs, afterLoadFunc)
		}

		for i := range t.NumField() {
			tt, vv := t.Field(i).Type, v.Field(i)

			// create local copy so tags are not modified between fields
			tags := tags
			if tag := t.Field(i).Tag.Get("weight"); tag != "" {
				// TODO: use model.Tag
				tags = append(tags, tag)
			}

			if tt == reflect.TypeOf((*mlx.Tensor)(nil)).Elem() {
				name := strings.Join(tags, ".")
				mapping[name] = append(mapping[name], vv.Addr().Interface().(*mlx.Tensor))
				continue
			}

			switch tt.Kind() {
			case reflect.Pointer, reflect.Struct:
				fn(vv, tags)
			case reflect.Slice, reflect.Array:
				for i := range vv.Len() {
					fn(vv.Index(i), append(tags, strconv.Itoa(i)))
				}
			}
		}
	}
	fn(reflect.ValueOf(m), []string{})
	return mapping, afterLoadFuncs
}

var m = make(map[string]func(*model.Root) (Model, error))

func Register(name string, f func(*model.Root) (Model, error)) {
	if _, exists := m[name]; exists {
		panic("model already registered: " + name)
	}

	m[name] = f
}

func New(root *model.Root) (Model, error) {
	c, err := root.Open("config.json")
	if err != nil {
		return nil, err
	}
	defer c.Close()

	var config struct {
		Architectures []string `json:"architectures"`
	}

	if err := json.NewDecoder(c).Decode(&config); err != nil {
		return nil, err
	}

	slog.Info("Model architecture", "arch", config.Architectures[0])
	if f, exists := m[config.Architectures[0]]; exists {
		return f(root)
	}

	return nil, errors.New("unknown architecture")
}
