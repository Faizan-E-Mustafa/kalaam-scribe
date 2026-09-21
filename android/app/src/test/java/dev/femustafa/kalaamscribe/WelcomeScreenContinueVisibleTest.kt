package dev.femustafa.kalaamscribe

import android.app.Application
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ApplicationProvider
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

/**
 * Regression guard for the welcome-screen bug: after picking a dictation language
 * the model list (~268dp) grows the fixed-height welcome column until the Continue
 * button falls below the fold on small phones (older code had no verticalScroll at
 * all). Each subclass renders on a phone-sized surface and must be able to scroll
 * the Continue button into view. Red on the old layout, green on the scrollable one.
 */
abstract class WelcomeScreenContinueBase {

    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun continueButtonIsReachableAfterPickingLanguage() {
        val app: Application = ApplicationProvider.getApplicationContext()
        val vm = ModelPickerViewModel(app)
        composeRule.setContent {
            LanguageOnboardingScreen(
                viewModel = vm,
                onSelect = {},
                onDownload = {},
                onLanguageChange = {},
                onContinue = {},
            )
        }

        // Pick a language so the model list renders and the button is enabled.
        composeRule.onNodeWithText("Language").performClick()
        composeRule.onNodeWithText("English (en)").performClick()

        // The whole screen must scroll: a fixed (non-scrollable) column would make
        // performScrollTo throw, and a clipped button would fail assertIsDisplayed.
        composeRule.onNodeWithText("Continue").performScrollTo().assertIsDisplayed()
    }
}

@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(qualifiers = "w411dp-h891dp-normal-long-notround")
class WelcomeScreenLargePhoneTest : WelcomeScreenContinueBase()

@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(qualifiers = "w360dp-h640dp-normal-notlong-notround")
class WelcomeScreenSmallPhoneTest : WelcomeScreenContinueBase()

@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(qualifiers = "w320dp-h480dp-normal-notlong-notround")
class WelcomeScreenTinyPhoneTest : WelcomeScreenContinueBase()