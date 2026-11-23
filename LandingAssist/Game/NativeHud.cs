using System;
using Il2CppMayday.UI;
using Il2CppTMPro;
using UnityEngine;
using UnityEngine.UI;

namespace MaydayLanding;

/// <summary>Landing panel built with the game's own UI factory (UIF), theme (UITheme) and display font,
/// docked under the native aircraft card (PlaneStatusWidget: TopLeft, pos 24,-24, size 236x262,
/// radius 16, UITheme.HudPanel). It follows that card's fade and slide, so it hides with the HUD.</summary>
internal sealed class NativeHud
{
    private const float Left = 24f, Width = 236f, CardBottom = -24f - 262f, Gap = 12f;
    private HudWidget? _anchor;
    private RectTransform? _root, _panelRoot, _body;
    private CanvasGroup? _group;
    private UIText? _title, _mode, _phase, _instruction, _detail, _controls, _recording;
    private readonly UIText?[] _labels = new UIText?[4], _values = new UIText?[4];
    private GameObject? _metricsRow1, _metricsRow2, _divider;

    public bool Active => _root != null && _root;

    /// <summary>Find the aircraft card and build the panel next to it; false keeps the IMGUI fallback.</summary>
    public bool Ensure()
    {
        var hud = UIRoot.Instance?.Hud;
        var card = hud?.planeStatus;
        if (card == null || card.Root == null) { Destroy(); return false; }
        if (Active && _anchor != null && _anchor.Pointer == card.Pointer) return true;
        Destroy();
        _anchor = card;
        var parent = card.Root.parent;
        _root = UIF.Rect("MaydayLandingPanel", parent);
        UIF.Stretch(_root, 0f, 0f, 0f, 0f);
        _group = _root.gameObject.AddComponent<CanvasGroup>();
        _group.blocksRaycasts = false;
        _group.interactable = false;

        var panel = UIF.Panel(_root, "Panel", UITheme.HudPanel, UITheme.Radius, true, false);
        _panelRoot = panel.Root;
        UIF.At(_panelRoot, UIF.TopLeft, new Vector2(Left, CardBottom - Gap), new Vector2(Width, 120f));

        _body = UIF.Rect("Body", _panelRoot);
        _body.anchorMin = new Vector2(0f, 1f);
        _body.anchorMax = new Vector2(1f, 1f);
        _body.pivot = new Vector2(0.5f, 1f);
        _body.anchoredPosition = Vector2.zero;
        _body.sizeDelta = Vector2.zero;
        UIF.VLayout(_body.gameObject, 4f, UIF.Pad(16, 16, 12, 14), TextAnchor.UpperLeft, true, true, true, false);
        UIF.Fit(_body.gameObject, false, true);

        var header = UIF.Row(_body, "Header", 24f, 8f, null);
        _title = UIF.Text(header, "Title", "进近辅助", 16f, Muted(), TextAlignmentOptions.Left, true, false);
        UIF.LE(_title.gameObject, -1f, -1f, 1f, -1f, -1f, -1f);
        _mode = UIF.Text(header, "Mode", "", 18f, UITheme.HudText, TextAlignmentOptions.Right, true, false);
        _phase = UIF.Text(_body, "Phase", "", 26f, UITheme.HudText, TextAlignmentOptions.Left, true, false);
        _instruction = UIF.Text(_body, "Instruction", "", 17f, UITheme.HudText, TextAlignmentOptions.Left, false, true);
        _divider = UIF.Divider(_body, UITheme.WithAlpha(UITheme.HudText, 0.12f), 2f).gameObject;
        _metricsRow1 = Metrics(_body, "Metrics1", 0);
        _metricsRow2 = Metrics(_body, "Metrics2", 2);
        _detail = UIF.Text(_body, "Detail", "", 15f, Muted(), TextAlignmentOptions.Left, false, true);
        UIF.Divider(_body, UITheme.WithAlpha(UITheme.HudText, 0.12f), 2f);
        _controls = UIF.Text(_body, "Controls", "", 15f, Muted(), TextAlignmentOptions.Left, false, true);
        _recording = UIF.Text(_body, "Recording", "", 15f, Muted(), TextAlignmentOptions.Left, false, true);
        return true;
    }

    private GameObject Metrics(RectTransform parent, string name, int first)
    {
        var row = UIF.Row(parent, name, 44f, 8f, null);
        for (int i = first; i < first + 2; i++)
        {
            var column = UIF.Rect("Cell" + i, row);
            UIF.VLayout(column.gameObject, 0f, UIF.Pad(0), TextAnchor.UpperLeft, true, true, true, false);
            UIF.LE(column.gameObject, -1f, -1f, 1f, -1f, -1f, -1f);
            _labels[i] = UIF.Text(column, "Label", "", 14f, Muted(), TextAlignmentOptions.Left, false, false);
            _values[i] = UIF.Text(column, "Value", "", 21f, UITheme.HudText, TextAlignmentOptions.Left, true, false);
        }
        return row.gameObject;
    }

    private static Color Muted() => UITheme.WithAlpha(UITheme.HudText, 0.62f);

    public static Color Tone(HudTone tone) => tone switch
    {
        HudTone.Engaged => UITheme.GreenLt,
        HudTone.Armed => UITheme.Gold,
        HudTone.Guide => UITheme.Sky,
        HudTone.Alert => UITheme.Red,
        _ => UITheme.HudText
    };

    public void Show(HudModel model, bool visible)
    {
        if (!Active) return;
        // Follow the aircraft card's fade and focus slide so both hide together.
        float alpha = 1f;
        var cardGroup = _anchor?.Group;
        if (cardGroup != null) alpha = cardGroup.alpha;
        _group!.alpha = visible ? alpha : 0f;
        if (_anchor?.Root != null) _root!.anchoredPosition = _anchor.Root.anchoredPosition;
        if (!visible) return;

        Set(_mode, model.Mode, Tone(model.ModeTone));
        Set(_phase, model.Phase, UITheme.HudText);
        Set(_instruction, model.Instruction, Tone(model.InstructionTone));
        for (int i = 0; i < 4; i++)
        {
            bool has = i < model.Metrics.Count;
            Set(_labels[i], has ? model.Metrics[i].Label : "", Muted());
            Set(_values[i], has ? model.Metrics[i].Value : "", has && model.Metrics[i].Warn ? UITheme.Red : UITheme.HudText);
        }
        if (_metricsRow1 != null) _metricsRow1.SetActive(model.Metrics.Count > 0);
        if (_metricsRow2 != null) _metricsRow2.SetActive(model.Metrics.Count > 2);
        if (_divider != null) _divider.SetActive(model.Metrics.Count > 0 || model.Detail.Length > 0);
        Set(_detail, model.Detail, Muted());
        Set(_controls, model.Controls, Muted());
        Set(_recording, model.Recording, model.RecordingActive ? UITheme.Red : Muted());
        if (_body != null && _panelRoot != null)
            _panelRoot.sizeDelta = new Vector2(Width, Mathf.Max(60f, _body.rect.height));
    }

    private static void Set(UIText? text, string value, Color color)
    {
        if (text == null) return;
        text.gameObject.SetActive(value.Length > 0);
        if (text.Text != value) text.Text = value;
        text.Color = color;
    }

    public void Destroy()
    {
        if (_root != null && _root) UnityEngine.Object.Destroy(_root.gameObject);
        _root = _panelRoot = _body = null;
        _group = null;
        _anchor = null;
        _title = _mode = _phase = _instruction = _detail = _controls = _recording = null;
        Array.Clear(_labels);
        Array.Clear(_values);
        _metricsRow1 = _metricsRow2 = _divider = null;
    }
}
